import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from openpoints.cpp.pointops.functions import pointops

_inf = 1e9
_eps = 1e-12

def inverse_sigmoid_function(cc, t, b):
    '''t = math.e'''
    '''a = 1 / (1 + np.power(t, cc))'''
    a = 1 / (1 + t.pow(b * cc))
    return a


'''https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py'''
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


