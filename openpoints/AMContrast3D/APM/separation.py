'''Ambiguity Prediction Module (APM)
   APM(p) -> a'''

from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models.build import MODELS
from openpoints.cpp.pointops.functions import pointops


@MODELS.register_module()
class APM_p(nn.Module):
    def __init__(self,
                 feature_dim: List[int] = [64, 128, 256, 512],
                 linear_mapping: bool = True,
                 cross_attention: bool = False,
                 feat_concate: bool = True,
                 channel: List[int] = [64, 32, 16, 8, 4, 2],
                 dropout: List[float] = [0.2, 0, 0, 0, 0, 0],
                 nsample_k: int = 12,
                 threshold: float = 0.7,
                 threshold_max: float = 1.0,
                 gamma: float = 0.5,
                 fusion: str = 'MIN',
                 att_dim: int = 3):
        super(APM_p, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, channel[0]),
            nn.Dropout(self.drop_rate[0]),
            nn.BatchNorm1d(channel[0]),
            nn.Sigmoid(),

            nn.Linear(channel[0], channel[1]),
            nn.Dropout(self.drop_rate[1]),
            nn.BatchNorm1d(channel[1]),
            nn.Sigmoid(),
   
            nn.Linear(channel[1], channel[2]),
            nn.Dropout(self.drop_rate[2]),
            nn.BatchNorm1d(channel[2]),
            nn.Sigmoid(),

            nn.Linear(channel[2], channel[3]),
            nn.Dropout(self.drop_rate[3]),
            nn.BatchNorm1d(channel[3]),
            nn.Sigmoid(),
 
            nn.Linear(channel[3], channel[4]),
            nn.Dropout(self.drop_rate[4]),
            nn.BatchNorm1d(channel[4]),
            nn.Sigmoid(),

            nn.Linear(channel[4], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, p):
        p = torch.flatten(p, start_dim=0, end_dim=1)  
        return self.layers(p)                        


def KNN(p, k):
    o = torch.unsqueeze(torch.tensor(p.shape[0], dtype=torch.int), 0).cuda()
    neighbor_idx, _ = pointops.knnquery(k, p, p, o, o)                           # [n, k]
    k -= 1  # exclude self-loop
    neighbor_idx = neighbor_idx[..., 1:].contiguous()                            # [n, k-1]
    m = neighbor_idx.shape[0]
    neighbor_pos = p[neighbor_idx.view(-1).long(), :].view(m, k, p.shape[1])       # [n, k-1, 3]
    return neighbor_idx, neighbor_pos


@MODELS.register_module()
class APM_p_Group(nn.Module):
    def __init__(self,
                 feature_dim: List[int] = [64, 128, 256, 512],
                 linear_mapping: bool = True,
                 cross_attention: bool = False,
                 feat_concate: bool = True,
                 channel: List[int] = [64, 32, 16, 8, 4, 2],
                 dropout: List[float] = [0.2, 0, 0, 0, 0, 0],
                 nsample_k: int = 12,
                 threshold: float = 0.7,
                 threshold_max: float = 1.0,
                 gamma: float = 0.5,
                 fusion: str = 'MIN',
                 att_dim: int = 3):
        super(APM_p_Group, self).__init__()

        self.k = nsample_k
        self.in_channels = nsample_k * 3
        self.conv = nn.Sequential(
            nn.Conv1d(self.in_channels, 18, kernel_size=1, bias=False),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.Conv1d(18, 9, kernel_size=1, bias=False),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Conv1d(9, 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU()
        )
        self.regressor = nn.Linear(3, 1)


    def forward(self, p):
        batch_size = p.shape[0]
        p = torch.flatten(p, start_dim=0, end_dim=1)                                 # [n=120000, 3]
        neighbor_idx, neighbor_pos = KNN(p, self.k)                                  # [n, k-1], [n, k-1, 3]
        p = torch.unsqueeze(p, 1)                                                    # [n, 1, 3]

        p_ij = torch.cat([p, torch.abs(p - neighbor_pos)], dim = 1)                  # [n, k, 3]:  [pi | pi-pj1, pi-pj2, ..., pi-pjk]
        p_ij = p_ij.view(p_ij.size(0), -1)                                           # [n, k*3]

        '''1. [mlp]'''
        # return self.mlp(p_ij)
        '''2. [conv]'''
        # p = p.view(batch_size, -1, 1, 3)                                           # [b, n/b, 1, 3]
        # neighbor_p = neighbor_p.view(batch_size, -1, self.k - 1, 3)                # [b, n/b, k-1, 3]
        p_ij = p_ij.view(batch_size, -1, self.k * 3)                                 # [b, n/b, k*3]
        p_ij = p_ij.transpose(1, 2)                                                  # [b, k*3, n/b]
        h = self.conv(p_ij)                                                          # [b, out_channel=3, n/b]
        h = h.transpose(1,2)
        out = self.regressor(h)

        out = F.softmax(out)
        out = torch.flatten(out, start_dim=0, end_dim=1)
        return out




class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x, adj, batch_size, k):
        '''x: [n, 1, 3]
         adj: [n, 3, k]'''
        x = torch.bmm(x, adj)                       # [n, 1, k]
        x = x.view(batch_size, -1, k)               # [b, n/b, k]
        x = x.transpose(1, 2)                       # [b, k, n/b]
        x = self.conv(x)
        return x

class POOL(nn.Module):
    def __init__(self):
        super(POOL, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        '''
        batch_size = x.size(0)
        x0 = self.max_pool(x).view(batch_size, -1)
        x1 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x0, x1), dim=-1)
        return x
        '''
        x_max = self.max_pool(torch.squeeze(x))
        x_avg = self.avg_pool(torch.squeeze(x))
        return x_max, x_avg


@MODELS.register_module()
class APM_p_Graph(nn.Module):
    def __init__(self,
                 feature_dim: List[int] = [64, 128, 256, 512],
                 linear_mapping: bool = True,
                 cross_attention: bool = False,
                 feat_concate: bool = True,
                 channel: List[int] = [64, 32, 16, 8, 4, 2],
                 dropout: List[float] = [0.2, 0, 0, 0, 0, 0],
                 nsample_k: int = 12,
                 threshold: float = 0.7,
                 threshold_max: float = 1.0,
                 gamma: float = 0.5,
                 fusion: str = 'MIN',
                 att_dim: int = 3):
        super(APM_p_Graph, self).__init__()

        self.k = nsample_k

        '''1. p*p_ij: [n,1,3]*[n,3,k]
        self.gcn = GCN(self.k, self.k, bias=False)
        self.bn = nn.BatchNorm1d(self.k)
        self.regressor = nn.Linear(self.k, 1)'''

        '''2. edge_index: each'''
        self.gcnconv = GCNConv(3, 1)
        self.pool = POOL()


    def forward(self, p):
        batch_size = p.shape[0]
        p = torch.flatten(p, start_dim=0, end_dim=1)                                 # [n=120000, 3]
        neighbor_idx, neighbor_pos = KNN(p, self.k)                                  # [n, k-1], [n, k-1, 3]
        p = torch.unsqueeze(p, 1)                                                    # [n, 1, 3]
        p_ij = torch.cat([p, torch.abs(p - neighbor_pos)], dim = 1)                  # [n, k, 3]:  [pi | pi-pj1, pi-pj2, ..., pi-pjk]

        '''1. p*p_ij: [n,1,3]*[n,3,k]
        p_ij = p_ij.transpose(1, 2)                                                  # [n, 3, k]
        h = F.leaky_relu(self.bn(self.gcn(p, p_ij, batch_size, self.k)), negative_slope=0.2)  # [b, k, n/b]
        h = h.transpose(1,2)                                                                  # [b, k, n/b]
        out = self.regressor(h)
        out = torch.flatten(out, start_dim=0, end_dim=1)
        print(out.shape)
        return out'''


        '''2. edge_index: each'''
        # node_feature = neighbor_pos   # [n, k-1, 3]  ### feature dim = 3
        node_feature = p_ij             # [n, k, 3]    ### feature dim = 3
        e = []
        e1 = []
        e2 = []
        for i in range(self.k-1):
            e1.append(0)
            e2.append(i+1)
        e.append(e1)
        e.append(e2)                                                      # [[0,0,...,0], [1,2,...,11]]
        edge_index = []
        for n in range(node_feature.shape[0]):
            edge_index.append(e)                                          # [[0,0,...,0], [1,2,...,11]] * n
        edge_index = torch.tensor(edge_index, dtype=torch.int64).cuda()   # [n, 2, k-1]

        print(self.gcnconv(node_feature, edge_index))
        node_embedding = torch.unsqueeze(self.gcnconv(node_feature[0], edge_index[0]), dim=0)        # [k, out_channel = 1] -> [1, k, 1]
        for n in range(1, node_feature.shape[0]):
            node_embedding_n = torch.unsqueeze(self.gcnconv(node_feature[n], edge_index[n]), dim=0)  # [1, k, 1]
            node_embedding = torch.cat([node_embedding, node_embedding_n], dim=0)                    # [n, k, 1]

        max_embedding, avg_embedding = self.pool(node_embedding)
        out = avg_embedding

        return out

        p = p.view(batch_size, -1, 3)                                                # [b, n/b, 3]
        p_ij = p_ij.view(batch_size, 3, -1)                                          # [b, 3, k*n/b]

 