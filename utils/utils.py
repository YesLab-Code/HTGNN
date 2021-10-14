import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


def compute_metric(pos_score, neg_score):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1))).detach().cpu()
    label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))
    pred_tag = torch.round(torch.sigmoid(pred))
    
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
#     acc = accuracy_score(label, pred_tag)
#     f1 = f1_score(label, pred_tag)

    return auc, ap


def compute_loss(pos_score, neg_score, device):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
    label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))
    
    return F.binary_cross_entropy_with_logits(pred, label.to(device))


def mp2vec_feat(path, g):
    wordvec = KeyedVectors.load(path, mmap='r')
    for ntype in g.ntypes:
        if ntype == 'author':
            prefix = 'a_'
        elif ntype == 'institution':
            prefix = 'i_'
        elif ntype == 'field_of_study':    
            prefix = 't_'
        else:
            break

        feat = torch.zeros(g.num_nodes(ntype),128)
        for j in range(g.num_nodes(ntype)):
            try:
                wv = np.array(wordvec[f'{prefix}{j}'])
                feat[j] = torch.from_numpy(wv)
            except KeyError:
#                 print(f'{prefix}{j}')
                continue
            
        g.nodes[ntype].data['feat'] = feat
    
    return g


