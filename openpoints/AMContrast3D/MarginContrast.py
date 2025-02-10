'''
AmbiguityHead: Return ai
ContrastHead: Return ai, loss
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.cpp.pointops.functions import pointops
from .AEF.utils import *
from .AEF.function import _eps
from .AEF.ambiguity import ambiguity_function

class AmbiguityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.nstride = torch.tensor([4, 4, 4, 4])
        self.ftype = get_ftype('latent')[0]
        self.posmask_func = getattr(self, f'posmask_cnt') 
        self.main = self.point_ambiguity

    def posmask_cnt(self, labels, neighbor_label):
        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)
        neighbor_label = torch.argmax(neighbor_label, -1)  
        mask = labels == neighbor_label
        return mask

    def point_ambiguity(self, n, i, stageACE_list, target, num_classes, ignore_index, ambiguity_args):
        p, features, o = fetch_pxo(n, i, stageACE_list, self.ftype)
        nsample = ambiguity_args.nsample
        labels = get_subscene_label_CBL(n, i, stageACE_list, target, self.nstride, num_classes, ignore_index)
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) 
        nsample -= 1 ### exclude self-loop
        neighbor_idx = neighbor_idx[..., 1:].contiguous()  
        m = neighbor_idx.shape[0]
        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1])
        posmask = self.posmask_cnt(labels, neighbor_label)
        point_mask = torch.sum(posmask.int(), -1) 
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)
        ambiguity_soft, _ = ambiguity_function(p, posmask, nsample, neighbor_idx, ambiguity_args.cctype, ambiguity_args.ccbeta, ambiguity_args.vis, ambiguity_args.nu)
        return ambiguity_soft

    def forward(self, target, stageACE_list, num_classes, ignore_index, ambiguity_args):
        target_ai_list = []
        stages = []
        for i in range(ambiguity_args.stages_num):
            stages.append((ambiguity_args.stages,i))
        for n, i in stages:
            target_ai = self.main(n, i, stageACE_list, target, num_classes, ignore_index, ambiguity_args)
            target_ai_list.append(target_ai)
        return target_ai_list



class ContrastHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.nstride = torch.tensor([4, 4, 4, 4])
        self.stages = [('up', 0), ('up', 1), ('up', 2), ('up', 3)]
        self.ftype = get_ftype('latent')[0]
        self.project = None
        '''Feature similarity: dist_kl; dist_l2; dist_dot; dist_cos'''
        self.dist_func = getattr(self, f'dist_cos')
        '''Contrastive Loss: contrast_softnn; contrast_nce; contrast_softnn_temperature; contrast_softnn_margin'''
        self.contrast_func = getattr(self, f'contrast_softnn_margin')
        self.posmask_func = getattr(self, f'posmask_cnt') 
        self.main_contrast = self.point_contrast_margin

    '''Direct Dot'''
    def dist_dot(self, features, neighbor_feature):
        dot = torch.mul(torch.unsqueeze(features, -2), neighbor_feature) 
        dist = torch.sum(dot, -1) + _eps
        return dist

    '''Cosine Similarity'''
    def dist_cos(self, features, neighbor_feature):
        dist = F.cosine_similarity(torch.unsqueeze(features, -2), neighbor_feature, dim=2)
        return dist

    '''Euclidean Distance'''
    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps) # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        # kl dist from featuers (gt) to neighbors (pred)
        if normalized in [False, 'softmax']:  # if still not a prob distribution - prefered
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')
        
        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        dist = dist.sum(-1)  # [m, nsample]
        return dist


    def posmask_cnt(self, labels, neighbor_label):
        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
        neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
        mask = labels == neighbor_label  # [m, nsample]
        return mask

    def contrast_softnn_margin(self, dist, posmask, ambiguity, ambiguity_args, invalid_mask=None):
        '''sum(sim(i, +)) / [sum(sim(i, +)) + sum(sim(i, -))]'''
        '''Ambiguity-aware Embedding'''
        if ambiguity_args.margin == 'constant':
            margin = ambiguity_args.nu
        elif ambiguity_args.margin == 'adaptive':
            u = ambiguity_args.mu
            v = ambiguity_args.nu
            margin = u * torch.unsqueeze(ambiguity, -1) + v

        elif ambiguity_args.margin == 'learned':
            u = torch.mean(dist * ~(posmask), 1)
            v = torch.mean(dist * posmask, 1)
            margin = (torch.unsqueeze(u, -1) - 1) * torch.unsqueeze(ambiguity, -1) + torch.unsqueeze(v, -1)
            print('cos_ik:', round(torch.max(u).item(),3), round(torch.min(u).item(),3))
            print('cos_ij:', round(torch.max(v).item(),3), round(torch.min(v).item(),3))
            print('m_i:', round(torch.max(margin).item(),3), round(torch.min(margin).item(),3))


        '''[EudDis] Decision Boundary: [-]-[+]>=m'''
        # dist = dist * posmask + (dist - margin) * ~(posmask)
        # dist = (dist + margin) * posmask + dist * ~(posmask)
        '''[CosSim] & [Dot] Decision Boundary: [+]-[-]>=m'''
        if ambiguity_args.db == '-m':
            dist = (dist - margin) * posmask + dist * ~(posmask)
        elif ambiguity_args.db == '+m':
            dist = dist * posmask + (dist + margin) * ~(posmask)
        else: 
            dist = dist * posmask + dist * ~(posmask)


        if ambiguity_args.temperature is not None:
            dist = dist / ambiguity_args.temperature
        exp = torch.exp(dist)
        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        '''
        pos: positive pairs for each boundary point
        neg: all (positive & negative) pairs for each boundary point
        '''
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp * (1 - posmask.int()), axis=-1) 
        pos_neg = torch.sum(exp, axis=-1) # (m)

        if ambiguity_args.supervisedCL == 'Method1':
            loss = pos / pos_neg + _eps
        elif ambiguity_args.supervisedCL == 'Method2':
            pos_ij = exp * posmask
            pos_ij_sumik = exp * posmask + neg.unsqueeze(-1)
            loss = pos_ij / pos_ij_sumik + _eps
       
            pos_num = torch.sum(posmask.int(), axis=-1) + _eps
            loss = torch.sum(loss, axis=-1) / pos_num 

        loss = - torch.log(loss)
        return loss


    def contrast_softnn_temperature(self, dist, posmask, ambiguity, ambiguity_args, invalid_mask=None):
        '''sum(pos) / sum(neg+pos)''' 
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if ambiguity_args.temperature is not None:
            dynamic_temperature = ambiguity_args.temperature * (1.5 - ambiguity)
            dist = dist / dynamic_temperature 
        exp = torch.exp(dist)
        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask
        '''
        pos: positive pairs for each boundary point
        neg: all (positive & negative) pairs for each boundary point
        '''
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = pos / neg + _eps
        loss = - torch.log(loss)
        return loss


    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        '''sum(pos) / sum(neg+pos)'''
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)
        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask
        '''
        pos: positive pairs for each boundary point
        neg: all (positive & negative) pairs for each boundary point
        '''
        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = pos / neg + _eps
        loss = - torch.log(loss)
        return loss


    def point_contrast_margin(self, n, i, stageACE_list, target, num_classes, ignore_index, ambiguity_args):
        p, features, o = fetch_pxo(n, i, stageACE_list, self.ftype)
        nsample = ambiguity_args.nsample
        labels = get_subscene_label_CBL(n, i, stageACE_list, target, self.nstride, num_classes, ignore_index)
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) # (m, nsample)    
        nsample -= 1 ### exclude self-loop
        neighbor_idx = neighbor_idx[..., 1:].contiguous()                          
        m = neighbor_idx.shape[0]
        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls) 
        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])
        posmask = self.posmask_cnt(labels, neighbor_label)  # (m, nsample) - bool
        point_mask = torch.sum(posmask.int(), -1)  # (m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)
        
        '''1. Ambiguity Calculation
                Hard: 0 & 1 & 0.5
                Soft: 0 & 1 & (0,1)
                ==> if n+ = 0: a = 1
                ==> if n+ = 23: a = 0
                ==> if 0 < n+ < 23: a = g(.)'''
        ambiguity_soft, _ = ambiguity_function(p, posmask, nsample, neighbor_idx, ambiguity_args.cctype, ambiguity_args.ccbeta, ambiguity_args.vis, ambiguity_args.nu)

        target_ai = torch.clone(ambiguity_soft)
        if 'ambiguity' in stageACE_list.keys():
            output_ai = stageACE_list['ambiguity'][i].flatten() #### Calculate \target_ai - output_ai\ for each layer 
        else:
            output_ai = None


        '''2. Loss Calculation'''
        point_mask_smallai = torch.logical_and(0 < ambiguity_soft, ambiguity_soft <= 1) 
        ambiguity_soft = ambiguity_soft[point_mask_smallai]
        posmask = posmask[point_mask_smallai]
        features = features[point_mask_smallai] 
        neighbor_feature = neighbor_feature[point_mask_smallai] 
        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask, ambiguity_soft, ambiguity_args)
        loss = torch.mean(loss) 

        return loss, output_ai, target_ai


    def forward(self, output, target, stageACE_list, num_classes, ignore_index, ambiguity_args):
        loss_sum = 0
        target_ai_list = []
        stages = [] ### self.stages: [('up', 0), ('up', 1), ('up', 2), ('up', 3)]
        for i in range(ambiguity_args.stages_num):
            stages.append((ambiguity_args.stages,i))
        for n, i in stages:
            loss, output_ai, target_ai = self.main_contrast(n, i, stageACE_list, target, num_classes, ignore_index, ambiguity_args)
            loss_sum += loss
            target_ai_list.append(target_ai)

        return loss_sum, torch.cat(target_ai_list), target_ai_list

        
