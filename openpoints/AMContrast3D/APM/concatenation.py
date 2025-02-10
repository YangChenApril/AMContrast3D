'''Ambiguity Prediction Module (APM)
   APM(ConCate(p,f)) -> a'''

from typing import List, Type
import torch
import torch.nn as nn
from openpoints.models.build import MODELS

@MODELS.register_module()
class APM_pf_ConCate(nn.Module):
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
        super(APM_pf_ConCate, self).__init__()
        self.dim = feature_dim
        self.map = linear_mapping
        self.drop_rate = dropout
        '''From <N*(3+D)> To <N*1>'''
        self.layer_0 = nn.Sequential(
            nn.Linear(3+feature_dim[0], channel[0]),
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
        self.layer_1 = nn.Sequential(
            nn.Linear(3+feature_dim[1], channel[0]),
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
        self.layer_2 = nn.Sequential(
            nn.Linear(3+feature_dim[2], channel[0]),
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
        self.layer_3 = nn.Sequential(
            nn.Linear(3+feature_dim[3], channel[0]),
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

        if self.map == True:
            '''From <N*1> To <N*D>'''
            self.map_0 = nn.Sequential(
                nn.Linear(1, feature_dim[0]),
                nn.Sigmoid()
            )
            self.map_1 = nn.Sequential(
                nn.Linear(1, feature_dim[1]),
                nn.Sigmoid()
            )
            self.map_2 = nn.Sequential(
                nn.Linear(1, feature_dim[2]),
                nn.Sigmoid()
            )
            self.map_3 = nn.Sequential(
                nn.Linear(1, feature_dim[3]),
                nn.Sigmoid()
            )

    def forward(self, p, f):
        if p.dim() == 2 and f.dim() == 2:
            x = torch.cat((p,f), dim=1)
        else:
            p = torch.flatten(p, start_dim=0, end_dim=1)  
            f = f.permute(0, 2, 1)                       
            f = torch.flatten(f, start_dim=0, end_dim=1)  
            x = torch.cat((p,f), dim=1)                   

        if f.shape[1] == self.dim[0]:
            if self.map == True:
                return self.layer_0(x), self.map_0(self.layer_0(x))
            else:
                return self.layer_0(x)
        elif f.shape[1] == self.dim[1]:
            if self.map == True:
                return self.layer_1(x), self.map_1(self.layer_1(x))
            else:
                return self.layer_1(x)
        elif f.shape[1] == self.dim[2]:
            if self.map == True:
                return self.layer_2(x), self.map_2(self.layer_2(x))
            else:
                return self.layer_2(x)
        elif f.shape[1] == self.dim[3]:
            if self.map == True:
                return self.layer_3(x), self.map_3(self.layer_3(x))
            else:
                return self.layer_3(x)