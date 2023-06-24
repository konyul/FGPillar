
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from .cbam import ZBAM
import torch_scatter
from det3d.utils.fuse import fold_conv_bn_eval_sequentialv2
class bin_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels, bn_folding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, in_channels//2, bias=False),
            nn.BatchNorm1d(in_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())
        self.bn_folding = bn_folding

    def forward(self, x):
        if self.bn_folding and not self.conv.training:
            self.conv = fold_conv_bn_eval_sequentialv2(self.conv)
            return self.conv(x)
        else:
            return self.conv(x)

class Zconv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.bin_shuffle = bin_shuffle((self.in_channels)*num_bins, out_channels)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        return src, occupied_mask

class conv1d(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.zbam = ZBAM(out_channels)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(out_channels/2), output_dim = out_channels, num_layers = 2)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        src = src.max(2)[0]
        return src, occupied_mask


class ZcCBAM(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        self.in_channels = model_cfg.input_channel
        self.out_channels = model_cfg.output_channel
        self.num_bins = model_cfg.num_bins
        self.zbam = ZBAM(model_cfg.output_channel, model_cfg.bn_folding)
        self.bin_shuffle = bin_shuffle((self.in_channels)*model_cfg.num_bins, model_cfg.output_channel, model_cfg.bn_folding)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(model_cfg.output_channel/2), output_dim = model_cfg.output_channel, num_layers = 2)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:,:,:4]
        src = src[:,:,1:]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        data_dict['mlp_feat'] = src
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        return src, occupied_mask

class HCBAM(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.in_channels = 32
        self.out_channels = 32
        self.num_bins = 32
        self.zbam = ZBAM(32, True)
        self.bin_shuffle = bin_shuffle((self.in_channels)*32, 32, True)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(32/2), output_dim = 32, num_layers = 2)
        self.bin_shuffle_t = bin_shuffle((5)*10, 32, True)
    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask
    def binning_t(self, data_dict):
        voxels, voxel_coords = data_dict['toxel_features'], data_dict['toxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['t_feat_unq_coords'], data_dict['t_feat_unq_inv'], data_dict['t_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], 10, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src_t, occupied_mask_t = self.binning_t(data_dict)
        occupied_mask = torch.logical_or(occupied_mask, occupied_mask_t)
    
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:,:,:4]
        src = src[..., 1:]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        N, Z, C = src.shape
        src = src.view(N, Z*C)
        src = self.bin_shuffle(src)
        src_t = src_t[occupied_mask]
        src_t = src_t[..., 1:].contiguous()
        N,T,C = src_t.shape
        src_t = src_t.view(N,T*C)
        src_t = self.bin_shuffle_t(src_t)
        src = src + src_t
    
        data_dict['mlp_feat'] = src
        return src, occupied_mask

def build_mlp(model_name='HCBAM'):
    model_dict = {
        'Zconv': Zconv,
        'CBAM': CBAM,
        'ZcCBAM': ZcCBAM,
        'HCBAM': HCBAM,
}
    model_class = model_dict[model_name]

    model = model_class(
                        )
    return model
