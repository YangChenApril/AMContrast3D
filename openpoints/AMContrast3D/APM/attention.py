'''Ambiguity Prediction Module (APM)
   APM(Attn(p,f)) -> a'''

from typing import List, Type
import torch
import torch.nn as nn
from openpoints.models.build import MODELS
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, 
                 in_dim: int=64,
                 dk: int=3,
                 dv: int=64):
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.query = nn.Linear(in_dim, dk, bias=False)
        self.key = nn.Linear(in_dim, dk, bias=False) 
        self.value = nn.Linear(in_dim, dv, bias=False)

    def forward(self, x, y):
        batch_size = x.shape[0] 
        '''[b, D, n] -> [b, n, D]'''            
        x = x.view(batch_size, -1, self.in_dim)
        y = y.view(batch_size, -1, self.in_dim)
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)
        '''Attention Score: [b * num_Q * num_K * num_K]'''
        QK = torch.matmul(Q, K.transpose(-2,-1)) / (self.in_dim ** 0.5) 
        QK_weights = F.softmax(QK, dim=-1)
        '''Output: [b * num_Q * num_K * out_dim]'''
        output = torch.bmm(QK_weights, V)
        return output


@MODELS.register_module()
class APM_pf_CrossAtt(nn.Module):
    def __init__(self, 
                 feature_dim: List[int] = [64, 128, 256, 512],
                 linear_mapping: bool = True,
                 cross_attention: bool = False,
                 feat_concate: bool = True,
                 channel: List[int] = [32, 16, 8, 4, 2],
                 dropout: List[float] = [0, 0, 0, 0, 0],
                 nsample_k: int = 12,
                 threshold: float = 0.7,
                 threshold_max: float = 1.0,
                 gamma: float = 0.5,
                 fusion: str = 'MIN',
                 att_dim: int = 3):
        super(APM_pf_CrossAtt, self).__init__()
        self.dim = feature_dim
        self.map = linear_mapping
        self.drop_rate = dropout
        self.mask_dim = att_dim

        '''From <N*D> To <N*1> '''      
        self.layer_0 = nn.Sequential(
            nn.Linear(feature_dim[0], channel[0]),
            nn.Sigmoid(),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
            nn.Linear(channel[1], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.layer_1 = nn.Sequential(
            nn.Linear(feature_dim[1], channel[0]),
            nn.Sigmoid(),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
            nn.Linear(channel[1], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(feature_dim[2], channel[0]),
            nn.Sigmoid(),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
            nn.Linear(channel[1], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(feature_dim[3], channel[0]),
            nn.Sigmoid(),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
            nn.Linear(channel[1], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.ext_0 = nn.Sequential(
            nn.Linear(3, feature_dim[0]),
            nn.Sigmoid()
        )
        self.ext_1 = nn.Sequential(
            nn.Linear(3, feature_dim[1]),
            nn.Sigmoid()
        )
        self.ext_2 = nn.Sequential(
            nn.Linear(3, feature_dim[2]),
            nn.Sigmoid()
        )
        self.ext_3 = nn.Sequential(
            nn.Linear(3, feature_dim[3]),
            nn.Sigmoid()
        )


    def forward(self, p, f):
        p = torch.flatten(p, start_dim=0, end_dim=1) 
        f = f.permute(0, 2, 1)                        
        f = torch.flatten(f, start_dim=0, end_dim=1)  

        cross_layer = Attention(f.shape[1], self.mask_dim, f.shape[1]).to('cuda')
        if f.shape[1] == self.dim[0]:
            x = cross_layer(self.ext_0(p), f)
            return self.layer_0(x)
        elif f.shape[1] == self.dim[1]:
            x = cross_layer(self.ext_1(p), f)
            return self.layer_1(x)
        elif f.shape[1] == self.dim[2]:
            x = cross_layer(self.ext_2(p), f)
            return self.layer_2(x)
        elif f.shape[1] == self.dim[3]:
            x = cross_layer(self.ext_3(p), f)
            return self.layer_3(x)


@MODELS.register_module()
class APM_pp_SelfAtt(nn.Module):
    def __init__(self, 
                 feature_dim: List[int] = [64, 128, 256, 512],
                 linear_mapping: bool = True,
                 cross_attention: bool = False,
                 feat_concate: bool = True,
                 channel: List[int] = [32, 16, 8, 4, 2],
                 dropout: List[float] = [0, 0, 0, 0, 0],
                 nsample_k: int = 12,
                 threshold: float = 0.7,
                 threshold_max: float = 1.0,
                 gamma: float = 0.5,
                 fusion: str = 'MIN',
                 att_dim: int = 3):
        super(APM_pp_SelfAtt, self).__init__()
        self.dim = feature_dim
        self.map = linear_mapping
        self.drop_rate = dropout
        self.mask_dim = att_dim
        self.layers = nn.Sequential(
            nn.Linear(3, channel[0]),
            nn.Sigmoid(),
            nn.Linear(channel[0], channel[1]),
            nn.Sigmoid(),
            nn.Linear(channel[1], 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, p):
        p = torch.flatten(p, start_dim=0, end_dim=1)
        cross_layer = Attention(p.shape[1], self.mask_dim, p.shape[1]).to('cuda')
        x = cross_layer(p, p)
        return self.layers(x)





