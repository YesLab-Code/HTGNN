import dgl
import torch
from utils.utils import mp2vec_feat

dgl.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

def construct_htg_covid(glist, idx, time_window):
    sub_glist = glist[idx-time_window:idx]

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)

    G_feat = dgl.heterograph(hetero_dict)
    
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']

    G_label = glist[idx]

    return G_feat, G_label


def load_COVID_data(glist, time_window):
    train_feats, train_labels = [], []
    val_feats, val_labels     = [], []
    test_feats, test_labels   = [], []

    for i in range(len(glist)):
        if i >= time_window:
            G_feat, G_label = construct_htg_covid(glist, i, time_window)
            if i >= len(glist)-30 and i <= len(glist)-1:
                test_feats.append(G_feat)
                test_labels.append(G_label)
            elif i >= len(glist)-60 and i <= len(glist)-30:
                val_feats.append(G_feat)
                val_labels.append(G_label)
            else:
                train_feats.append(G_feat)
                train_labels.append(G_label)
    
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels


def construct_htg_mag(glist, idx, time_window):
    sub_glist = glist[idx-time_window:idx]

    ID_dict = {}

    for ntype in glist[0].ntypes:
        ID_set = set()
        for g_s in sub_glist:
            tmp_set = set(g_s.ndata['_ID'][ntype].tolist())
            ID_set.update(tmp_set)
        ID_dict[ntype] = {ID: idx for idx, ID in enumerate(sorted(list(ID_set)))}

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            ID_src = g_s.ndata['_ID'][srctype]
            ID_dst = g_s.ndata['_ID'][dsttype]
            new_src = ID_src[src]
            new_dst = ID_dst[dst]

            new_new_src = [ID_dict[srctype][e.item()] for e in new_src]
            new_new_dst = [ID_dict[dsttype][e.item()] for e in new_dst]

            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (new_new_src, new_new_dst)
            hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (new_new_dst, new_new_src)

    G_feat = dgl.heterograph(hetero_dict)
    
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            G_feat.nodes[ntype].data[f't{t}'] = torch.zeros(G_feat.num_nodes(ntype), g_s.nodes[ntype].data['feat'].shape[1])
            node_id = g_s.ndata['_ID'][ntype]
            node_feat = g_s.ndata['feat'][ntype]
            for (id, feat) in zip(node_id, node_feat):
                G_feat.nodes[ntype].data[f't{t}'][ID_dict[ntype][id.item()]] = feat

    return G_feat


def generate_APA(graph, device):
    AP = graph.adj(etype=('author', 'writes', 'paper')).to_dense()
    PA = AP.t()
    APA = torch.mm(AP.to(device), PA.to(device)).detach().cpu()
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    
    return APA


def construct_htg_label_mag(glist, idx, device):

    APA_cur = generate_APA(glist[idx], device)
    APA_pre = generate_APA(glist[idx-1], device)
    
    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()
    
    APA_sub = APA_cur - APA_pre # new co-author relation
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5
    
    # get indices of author pairs who collaborate
    indices_true = (APA_sub == 1).nonzero(as_tuple=True)
    indices_false = (APA_add == 0).nonzero(as_tuple=True)
    
    pos_src = indices_true[0]
    pos_dst = indices_true[1]
    
    size = int(pos_src.shape[0] * 0.1)
    
    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    pos_src = pos_src[pos_idx]
    pos_dst = pos_dst[pos_idx] 
    
    neg_src = indices_false[0]
    neg_dst = indices_false[1]

    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    neg_src = neg_src[neg_idx]
    neg_dst = neg_dst[neg_idx]
    
    return dgl.graph((pos_src, pos_dst), num_nodes=APA_cur.shape[0]), dgl.graph((neg_src, neg_dst), num_nodes=APA_cur.shape[0])


def load_MAG_data(glist, time_window, device):
    
    print('loading mp2vec')
    glist = [mp2vec_feat(f'mp2vec/g{i}.vector', g) for (i, g) in enumerate(glist)]

    train_feats, train_labels = [], []
    val_feats, val_labels     = [], []
    test_feats, test_labels   = [], []

    print(f'generating train, val, test sets ')
    for i in range(len(glist)):
        if i >= time_window:
            G_feat = construct_htg_mag(glist, i, time_window)
            pos_label, neg_label = construct_htg_label_mag(glist, i, device)
            if i == len(glist)-1:
                test_feats.append(G_feat)
                test_labels.append((pos_label, neg_label))
            elif i == len(glist)-2:
                val_feats.append(G_feat)
                val_labels.append((pos_label, neg_label))
            else: 
                train_feats.append(G_feat)
                train_labels.append((pos_label, neg_label))
                
    return train_feats, train_labels, val_feats, val_labels, test_feats, test_labels