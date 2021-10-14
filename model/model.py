import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from dgl.nn.pytorch import GATConv
import numpy as np
import math


class RelationAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        """

        :param n_inp: int, input dimension
        :param n_hid: int, hidden dimension
        """
        super(RelationAgg, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid),
            nn.Tanh(),
            nn.Linear(n_hid, 1, bias=False)
        )

    def forward(self, h):
        w = self.project(h).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)

        return (beta * h).sum(1)

    
class TemporalAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int, time_window: int, device: torch.device):
        """

        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param time_window: int         , the number of timestamps
        :param device     : torch.device, gpu
        """
        super(TemporalAgg, self).__init__()

        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.k_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.v_w  = nn.Linear(n_hid, n_hid, bias = False)
        self.fc   = nn.Linear(n_hid, n_hid)
        self.pe   = torch.tensor(self.generate_positional_encoding(n_hid, time_window)).float().to(device)

    def generate_positional_encoding(self, d_model, max_len):
        pe = np.zeros((max_len, d_model))
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * - math.log(100000.0) / d_model)
                pe[i][k] = math.sin( (i + 1) * div_term)
                try:
                    pe[i][k + 1] = math.cos( (i + 1) * div_term)
                except:
                    continue
        return pe
   
    def forward(self, x):
        x = x.permute(1,0,2)
        h = self.proj(x)
        h = h + self.pe
        q = self.q_w(h)
        k = self.k_w(h)
        v = self.v_w(h)
       
        qk = torch.matmul(q, k.permute(0, 2, 1))
        score = F.softmax(qk, dim = -1)

        h_ = torch.matmul(score, v)
        h_ = F.relu(self.fc(h_))
        
        return h_


class HTGNNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_heads: int, timeframe: list, norm: bool, device: torch.device, dropout: float):
        """

        :param graph    : dgl.DGLGraph, a heterogeneous graph
        :param n_inp    : int         , input dimension
        :param n_hid    : int         , hidden dimension
        :param n_heads  : int         , number of attention heads
        :param timeframe: list        , list of time slice
        :param norm     : bool        , use LayerNorm or not
        :param device   : torch.device, gpu
        :param dropout  : float       , dropout rate
        """
        super(HTGNNLayer, self).__init__()
        
        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_heads   = n_heads
        self.timeframe = timeframe
        self.norm      = norm
        self.dropout   = dropout

        # intra reltion aggregation modules
        self.intra_rel_agg = nn.ModuleDict({
            etype: GATConv(n_inp, n_hid, n_heads, feat_drop=dropout, allow_zero_in_degree=True)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        # inter relation aggregation modules
        self.inter_rel_agg = nn.ModuleDict({
            ttype: RelationAgg(n_hid, n_hid)
            for ttype in timeframe
        })
        
        # inter time aggregation modules
        self.cross_time_agg = nn.ModuleDict({
            ntype: TemporalAgg(n_hid, n_hid, len(timeframe), device)
            for ntype in graph.ntypes
        })
        
        # gate mechanism
        self.res_fc = nn.ModuleDict()
        self.res_weight = nn.ParameterDict()
        for ntype in graph.ntypes:
            self.res_fc[ntype] = nn.Linear(n_inp, n_heads * n_hid)
            self.res_weight[ntype] = nn.Parameter(torch.randn(1))

        self.reset_parameters()
        
        # LayerNorm
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: dgl.DGLGraph, node_features: dict):
        """

        :param graph: dgl.DGLGraph
        :param node_features: dict, {'ntype': {'ttype': features}}
        :return: output_features: dict, {'ntype': {'ttype': features}}
        """

        # same type neighbors aggregation
        # intra_features, dict, {'ttype': {(stype, etype, dtype): features}}
        intra_features = dict({ttype:{} for ttype in self.timeframe})

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            reltype = etype.split('_')[0]
            ttype = etype.split('_')[-1]
            dst_feat = self.intra_rel_agg[etype](rel_graph, (node_features[stype][ttype], node_features[dtype][ttype]))
            # dst_representation (dst_nodes, hid_dim)
            intra_features[ttype][(stype, etype, dtype)] = dst_feat.squeeze()
        
        # different types aggregation
        # inter_features, dict, {'ntype': {ttype: features}}
        inter_features = dict({ntype:{} for ntype in graph.ntypes})
        
        for ttype in intra_features.keys():
            for ntype in graph.ntypes:
                types_features = []
                for stype, etype, dtype in intra_features[ttype]:
                    if ntype == dtype:
                        types_features.append(intra_features[ttype][(stype, etype, dtype)])

                types_features = torch.stack(types_features, dim=1)
                out_feat = self.inter_rel_agg[ttype](types_features)
                inter_features[ntype][ttype] = out_feat
            
        # different timestamps aggregation
        # time_features, dict, {'ntype': {'ttype': features}}
        output_features = {}
        for ntype in inter_features:
            output_features[ntype] = {}
            out_emb = [inter_features[ntype][ttype] for ttype in inter_features[ntype]]
            time_embeddings = torch.stack(out_emb, dim=0)
            h = self.cross_time_agg[ntype](time_embeddings).permute(1,0,2)
            output_features[ntype] = {ttype: h[i] for (i, ttype) in enumerate(self.timeframe)}
        
        new_features = {}
        for ntype in output_features:
            new_features[ntype] = {}
            alpha = torch.sigmoid(self.res_weight[ntype])
            for ttype in self.timeframe:
                new_features[ntype][ttype] = output_features[ntype][ttype] * alpha + self.res_fc[ntype](node_features[ntype][ttype]) * (1 - alpha)
                if self.norm:
                    new_features[ntype][ttype] = self.norm_layer[ntype](new_features[ntype][ttype])

        return new_features


class HTGNN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_inp: int, n_hid: int, n_layers: int, n_heads: int, time_window: int, norm: bool, device: torch.device, dropout: float = 0.2):
        """

        :param graph      : dgl.DGLGraph, a dgl heterogeneous graph
        :param n_inp      : int         , input dimension
        :param n_hid      : int         , hidden dimension
        :param n_layers   : int         , number of stacked layers
        :param n_heads    : int         , number of attention heads
        :param time_window: int         , number of timestamps
        :param norm       : bool        , use LayerNorm or not
        :param device     : torch.device, gpu
        :param dropout    : float       , dropout rate
        """
        super(HTGNN, self).__init__()

        self.n_inp     = n_inp
        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.timeframe = [f't{_}' for _ in range(time_window)]

        self.adaption_layer = nn.ModuleDict({ntype: nn.Linear(n_inp, n_hid) for ntype in graph.ntypes})
        self.gnn_layers     = nn.ModuleList([HTGNNLayer(graph, n_hid, n_hid, n_heads, self.timeframe, norm, device, dropout) for _ in range(n_layers)])

    def forward(self, graph: dgl.DGLGraph, predict_type: str):
        """

        :param graph       : dgl.DGLGraph, a dgl heterogeneous graph
        :param predict_type: str         , predicted node type
        """

        # node feature adaption
        # inp_feat: dict, {'ntype': {'ttype': features}}
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = {}
            for ttype in self.timeframe:
                inp_feat[ntype][ttype] = self.adaption_layer[ntype](graph.nodes[ntype].data[ttype])

        # gnn
        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](graph, inp_feat)

        out_feat = sum([inp_feat[predict_type][ttype] for ttype in self.timeframe])
        
        return out_feat
    

class LinkPredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """
    
        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()
        self.fc1 = nn.Linear(n_inp * 2, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.fc2(F.relu(self.fc1(x)))
        return {'score': y}

    def forward(self, graph: dgl.DGLGraph, node_feat: torch.tensor):
        """
        
        :param graph    : dgl.DGLGraph  
        :param node_feat: torch.tensor
        """
        with graph.local_scope():
            graph.ndata['h'] = node_feat
            graph.apply_edges(self.apply_edges)
        
            return graph.edata['score']


class NodePredictor(nn.Module):
    def __init__(self, n_inp: int, n_classes: int):
        """
    
        :param n_inp      : int, input dimension
        :param n_classes  : int, number of classes
        """
        super().__init__()

        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 =  nn.Linear(n_inp, n_classes)

    def forward(self, node_feat: torch.tensor):
        """
        
        :param node_feat: torch.tensor
        """

        node_feat = F.relu(self.fc1(node_feat))
        pred = F.relu(self.fc2(node_feat))
        
        return pred
