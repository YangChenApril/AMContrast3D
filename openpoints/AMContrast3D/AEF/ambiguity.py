import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from openpoints.cpp.pointops.functions import pointops
from .function import _eps, inverse_sigmoid_function, square_distance
from ..vis import vis_points


def ambiguity_function(p, posmask, nsample, neighbor_idx, ambiguity_type, ambiguity_beta, ambiguity_vis, nu):
    '''23 & 0 & (0,23) --> 0 (inner) & 1 (wrong) & (0,1) (boundary)'''
    mask_num = torch.sum(posmask.int(), -1) ### m 
    ambiguity_value_soft = torch.abs(mask_num.add(-torch.max(mask_num))).div(torch.max(mask_num)) ### |mask_num-23|/23
    idx_boundary = torch.logical_and(0 < ambiguity_value_soft, ambiguity_value_soft < 1)

    '''=====> n+ n-'''
    mask_boundary = torch.logical_and(0 < mask_num, mask_num < torch.max(mask_num)) # (m)
    mask_b = posmask[mask_boundary] # (m_b, 23) True & False
    n_pos = torch.sum(mask_b.int(), -1)  # (m_b)
    n_neg = torch.sum(1-mask_b.int(), -1) # (m_b)
 
    '''=====> d+ d-'''
    if ambiguity_type == 'Method1':
        d_pos = torch.full(n_pos.shape, 5.0).cuda()
        d_neg = torch.full(n_neg.shape, 5.0).cuda()
    elif ambiguity_type == 'Method2':
        BNC = torch.unsqueeze(p[idx_boundary], 1).cuda() ### (B, C) --> (B, N, C): N=1 each boundary point, C=3 xn yn zn coords
        neighbor_idx_boundary = neighbor_idx[idx_boundary] ### (B, M)
        BMC = torch.index_select(p, 0, neighbor_idx_boundary[0]).cuda() ### (M, C)
        BMC = torch.unsqueeze(BMC, 0) ### (M, C) --> (B, M, C): B=1 at first then B++, M=23 neighbors, C=3 xm ym zm coords
        for b in range(1, neighbor_idx_boundary.shape[0]):
            BMC_add = torch.index_select(p, 0, neighbor_idx_boundary[b])
            BMC_add = torch.unsqueeze(BMC_add, 0)
            BMC = torch.cat((BMC, BMC_add), 0)
        dd = square_distance(BNC, BMC)
        d_pos = torch.sum(mask_b.int() * dd.squeeze(), -1)  # (m_b)
        d_neg = torch.sum((1-mask_b.int()) * dd.squeeze(), -1)  # (m_b)
    elif ambiguity_type == 'Method3':
        BNC = torch.unsqueeze(p[idx_boundary], 1).cuda() ### (B, C) --> (B, N, C): N=1 each boundary point, C=3 xn yn zn coords
        neighbor_idx_boundary = neighbor_idx[idx_boundary] ### (B, M)
        BMC = torch.index_select(p, 0, neighbor_idx_boundary[0]).cuda() ### (M, C)
        BMC = torch.unsqueeze(BMC, 0) ### (M, C) --> (B, M, C): B=1 at first then B++, M=23 neighbors, C=3 xm ym zm coords
        for b in range(1, neighbor_idx_boundary.shape[0]):
            BMC_add = torch.index_select(p, 0, neighbor_idx_boundary[b])
            BMC_add = torch.unsqueeze(BMC_add, 0)
            BMC = torch.cat((BMC, BMC_add), 0)
        dd = square_distance(BNC, BMC) 
        '''====> Root of Square'''
        dd = torch.sqrt(torch.abs(dd) + _eps)
        d_pos = torch.sum(mask_b.int() * dd.squeeze(), -1)  # (m_b)
        d_neg = torch.sum((1-mask_b.int()) * dd.squeeze(), -1)  # (m_b)


    '''=====> cc+ cc-'''
    cc_pos = n_pos / d_pos
    cc_neg = n_neg / d_neg

    '''=====> a = f(n+,n-,d+,d-)'''
    t = torch.full(cc_pos.shape, math.e).cuda()
    ai_soft = inverse_sigmoid_function(cc_pos - cc_neg, t, ambiguity_beta) 

    #   0=======================0
    #   |   3. Visualization    |
    #   0=======================0
    if ambiguity_vis == True:
        print('Hard Value Visualization:')
        ambiguity_value_soft.scatter_(0, torch.nonzero(idx_boundary==True).squeeze(), nu)
        vis_points(points=p, colors=None, labels=ambiguity_value_soft)
        
    ambiguity_value_soft.scatter_(0, torch.nonzero(idx_boundary==True).squeeze(), ai_soft)

    if ambiguity_vis == True:
        print('Soft Value Visualization:')
        vis_points(points=p, colors=None, labels=ambiguity_value_soft) 

    nu_m = nu * 10

    ambiguity_0 = ambiguity_value_soft == 0
    ambiguity_low = torch.logical_and(0 < torch.ceil(ambiguity_value_soft*10), torch.ceil(ambiguity_value_soft*10) < nu_m) # nu_m = 5 or 6
    ambiguity_semi = torch.ceil(ambiguity_value_soft*10) == nu_m
    ambiguity_high = torch.logical_and(nu_m < torch.ceil(ambiguity_value_soft*10), torch.ceil(ambiguity_value_soft*10) < 10)
    ambiguity_1 = torch.ceil(ambiguity_value_soft*10) == 10

    count_0 = round(torch.sum(ambiguity_0).item() / len(ambiguity_value_soft) * 100, 2)
    count_low = round(torch.sum(ambiguity_low).item() / len(ambiguity_value_soft) * 100, 2)
    count_semi = round(torch.sum(ambiguity_semi).item() / len(ambiguity_value_soft) * 100, 2)
    count_high = round(torch.sum(ambiguity_high).item() / len(ambiguity_value_soft) * 100, 2)
    count_1 = round(torch.sum(ambiguity_1).item() / len(ambiguity_value_soft) * 100, 2)

    count_low_semi_high = [count_0, count_low, count_semi, count_high, count_1]

    return ambiguity_value_soft, count_low_semi_high



