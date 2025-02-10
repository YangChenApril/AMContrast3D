import torch
import torch.nn as nn
from torch.functional import F
from openpoints.cpp.pointops.functions import pointops
from openpoints.AMContrast3D.APM.attention import Attention

class RefinementMethod():

    def __init__(self, stage_list, p, f, a, i, B, K, fusion, threshold_max, threshold, gamma):
        self.stage_list = stage_list
        self.position = p
        self.feature = f
        self.ambiguity = a
        self.i = i
        self.batch = B
        self.sample_k = K
        self.fusion = fusion
        self.threshold_max = threshold_max
        self.threshold = threshold
        self.gamma = gamma

    def MapAttention(self):
        dim = self.stage_list['ambiguity_map'][self.i].shape[1]
        a_map = self.stage_list['ambiguity_map'][self.i].unsqueeze(0).view(self.batch, dim, -1)   ### [N, D] -> [b, D, n]
        if self.i == -1:
            '''x=Q; y=V,K'''
            dim_mask = 3
            cross_layer = Attention(dim, dim_mask, dim).to('cuda')  
            self.feature = cross_layer(x=a_map, y=self.feature)           ### [b, n, D]
            self.feature = self.feature.view(self.batch, dim, -1).cuda()  ### [b, D, n]
        return self.feature

    def MapSum(self):
        dim = self.stage_list['ambiguity_map'][self.i].shape[1]
        a_map = self.stage_list['ambiguity_map'][self.i].unsqueeze(0).view(self.batch, dim, -1)   ### [N, D] -> [b, D, n]
        self.feature = self.feature + a_map                  
        return self.feature

    def MapMultiply(self):
        dim = self.stage_list['ambiguity_map'][self.i].shape[1]
        a_map = self.stage_list['ambiguity_map'][self.i].unsqueeze(0).view(self.batch, dim, -1)   ### [N, D] -> [b, D, n]
        self.feature = torch.mul(self.feature, a_map)         
        return self.feature

    def Multiply(self):
        self.feature = torch.mul(self.feature, self.ambiguity)                 
        return self.feature

    def DualMasks(self):
        xyz = self.position.view(-1, 3)                                      ### [b*n, 3]
        o = torch.IntTensor([xyz.shape[0]]).cuda()                           ### [b*n]
        neighbor_idx, _ = pointops.knnquery(self.sample_k, xyz, xyz, o, o)   ### [b*n, K]
        D = self.feature.shape[1]
        f_knn = self.feature.view(-1, D)  
        a_knn = self.ambiguity.view(-1, 1)  
        # print(self.position.shape, xyz.shape)                 ### [b, n, 3], [b*n, 3]
        # print(self.feature.shape, f_knn.shape)                ### [b, D, n], [b*n, D]
        # print(self.ambiguity.shape, a_knn.shape)              ### [b, 1, n], [b*n, 1]
        self.sample_k -= 1                                      ### exclude self-loop
        neighbor_idx = neighbor_idx[..., 1:].contiguous()                        
        m = neighbor_idx.shape[0]
        neighbor_feature = f_knn[neighbor_idx.view(-1).long(), :].view(m, self.sample_k, f_knn.shape[1])   # [b*n, K-1, D]
        neighbor_ambiguity = a_knn[neighbor_idx.view(-1).long(), :].view(m, self.sample_k, a_knn.shape[1]) # [b*n, K-1, 1]

        '''(1) CrossMask: Searching neighboring points with min(ambiguity).'''
        CrossMask = self.cross_mask(neighbor_ambiguity, neighbor_feature, D)

        '''(2) SelfMask: Defining high-ai points with ambiguity >= threshold.
               True=1:  ai >= threshold ==> update
               False=0: ai < threshold  ==> remain'''
        SelfMask, rate = self.self_mask()

        '''(3) Updating Rate
               Constant: gamma = 1   (update all high-ambiguity points) 
               Constant: gamma = 0.5 (mean of f[i-1] and f_new)
               Constant: gamma = 0   (remain f[i-1] = f[i-1])
               Adaptive: gamma_new = threshold_max - a '''
        f_new = self.feature * ~SelfMask  + CrossMask * SelfMask

        '''Constant gamma'''
        self.feature = self.gamma * f_new + (1 - self.gamma) * self.feature        ### <Constant gamma>
        '''Apaptive gamma  
        gamma_new = self.threshold_max - self.ambiguity                            ### <Adaptive gamma>
        self.feature = gamma_new * f_new + (1 - gamma_new) * self.feature'''

        return self.feature, rate


    def cross_mask(self, neighbor_ambiguity, neighbor_feature, D):
        if self.fusion == 'MIN':
            good_ai = torch.min(neighbor_ambiguity, 1).values      # [b*n, 1]
            good_idx = torch.min(neighbor_ambiguity, 1).indices    # [b*n, 1]   ### good_idx.view(-1).shape = b*n 

            neighbor_ambiguity_bool = neighbor_ambiguity.squeeze() * 0.0                                         ### [b*n, K]  only 0.0
            neighbor_ambiguity_bool = neighbor_ambiguity_bool.scatter_(1, good_idx, 1.0)                         ### [b*n, K]  0.0 and 1.0
            neighbor_ambiguity_bool_D = torch.stack([neighbor_ambiguity_bool, neighbor_ambiguity_bool], axis=2)  ### [b*n, K, 2] 
            for d in range(D-2):
                neighbor_ambiguity_bool_D = torch.cat([neighbor_ambiguity_bool_D, torch.unsqueeze(neighbor_ambiguity_bool, 2)], axis=2)  ### [b*n, K, D]                                       
            good_feat = neighbor_feature * neighbor_ambiguity_bool_D                        ### [b*n, K, D]
            good_feat = torch.sum(good_feat, dim=1)  
            good_feat = good_feat.view(self.feature.shape[0], D, -1)                                    
        elif self.fusion == 'MIN_ALL0':
            # print(neighbor_ambiguity.ge(0))                          ### [b*n, K, 1]
            # print(neighbor_ambiguity.gt(0))                          ### [b*n, K, 1]
            good_feat = neighbor_feature * ~neighbor_ambiguity.gt(0)   ### [b*n, K, D]
            # good_feat = torch.sum(good_feat, dim=1)  ### SUM
            good_feat = torch.mean(good_feat, dim=1) ### MEAN
            good_feat = good_feat.view(self.feature.shape[0], D, -1) 
        return good_feat
        

    def self_mask(self):
        mask_1 = self.ambiguity.le(self.threshold_max)   ### ai <= Epsilon_2
        mask_2 = self.ambiguity.ge(self.threshold)       ### ai >= Epsilon_1
        mask = mask_1 * mask_2     

        update_count = torch.count_nonzero(mask.long()).item()
        update_percent = (update_count / self.ambiguity.numel()) * 100
        return mask, update_percent


    def consistency_regularization(self, embedding_1, embedding_2):
        '''Jensen-Shannon divergence'''
        net_1_probs = F.softmax(embedding_1, dim=0)
        net_2_probs = F.softmax(embedding_2, dim=0)
        
        total_m = 0.5 * (net_1_probs + net_2_probs)
        
        loss = 0.0
        loss += F.kl_div(F.log_softmax(embedding_1, dim=0), total_m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(embedding_2, dim=0), total_m, reduction="batchmean") 
        return (0.5 * loss)