import torch
import torch.nn as nn
from typing import List, Tuple
try:
    import spconv.pytorch as spconv
except:
    import spconv
from .pillar_utils import bev_spatial_shape, PillarQueryAndGroup, FGPillarQueryAndGroup
from .scatter_utils import scatter_max
from det3d.models.utils.mlp import build_mlp
from det3d.models.utils.dsp import build_dsp
import torch_scatter

class PillarMaxPooling(nn.Module):
    def __init__(self, mlps: List[int], pillar_size:float, point_cloud_range:List[float]):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, pillar_size)
        self.groups = PillarQueryAndGroup(pillar_size, point_cloud_range)

        shared_mlp = []
        for k in range(len(mlps) - 1):
            shared_mlp.extend([
                nn.Linear(mlps[k], mlps[k + 1], bias=False),
                nn.BatchNorm1d(mlps[k + 1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])
        self.shared_mlps = nn.Sequential(*shared_mlp)

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, pt_feature):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]
        pillar_indices, pillar_set_indices, group_features = self.groups(xyz, xyz_batch_cnt, pt_feature)

        group_features = self.shared_mlps(group_features)  # (1, C, L)
        group_features = group_features.transpose(1, 0).contiguous()

        pillar_features = scatter_max(group_features, pillar_set_indices, pillar_indices.shape[0])   # (C, M)
        pillar_features = pillar_features.transpose(1, 0)   # (M, C)

        return spconv.SparseConvTensor(pillar_features, pillar_indices, (self.bev_height, self.bev_width), B)


class FGPillarMaxPooling(nn.Module):
    def __init__(self, mlps: List[int], pillar_size:float, point_cloud_range:List[float],
                 dsp_cfg=None, zpillar_cfg=None):
        super().__init__()

        self.bev_width, self.bev_height = bev_spatial_shape(point_cloud_range, pillar_size)
        self.groups = FGPillarQueryAndGroup(pillar_size, point_cloud_range)

        self.init_weights(weight_init='xavier')
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        if dsp_cfg is not None:
            self.dsp_cfg = dsp_cfg
            pillar_size = dsp_cfg.voxel_size[0]
            self.voxel_size = torch.tensor([pillar_size,pillar_size,0.25]).cuda()
            grid_size = torch.tensor(dsp_cfg.grid_size).cuda()
            self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
            self.scale_yz = grid_size[1] * grid_size[2]
            self.scale_xy = grid_size[0] * grid_size[1]
            self.scale_y = grid_size[1]
            self.scale_z = grid_size[2]
            self.dsp_model = build_dsp(dsp_cfg)
        if zpillar_cfg is not None:
            self.zpillar_cfg = zpillar_cfg
            self.zpillar_model = build_mlp(zpillar_cfg, model_name='ZcCBAM')
        

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def dyn_voxelization(self, points, point_coords, example):
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        example['voxel_features'] = points_mean.contiguous()
        example['voxel_features_coords'] = voxel_coords.contiguous()
        return example
    
    def forward(self, xyz, xyz_batch_cnt, pt_feature, bxyz, example):
        """
        Args:
            xyz: (N1+N2..., 3)
            xyz_batch_cnt:  (N1, N2, ...)
            point_features: (N1+N2..., C)
            spatial_shape: [B, H, W]
        Return:
            pillars: (M1+M2..., 3) [byx]
            pillar_features: (M, C)
        """
        B = xyz_batch_cnt.shape[0]
        pillar_indices, pillar_set_indices, group_features, group_point_features, group_point_bxyz, group_pillar_centers = self.groups(xyz, xyz_batch_cnt, pt_feature, bxyz)
        example['points_with_f_center'] = group_point_features

        example['points'] = group_point_bxyz
        pillar_set_indices = pillar_set_indices.long()
        example['unq_inv'] = pillar_set_indices
        if self.zpillar_cfg is not None:            
            points_coords_3d = torch.floor((group_point_bxyz[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
            example = self.dyn_voxelization(group_point_bxyz, points_coords_3d, example)
            voxel_features_coords = example['voxel_features_coords']
            v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:, 3] * self.scale_y + voxel_features_coords[:, 2]
            v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
            example['v_feat_unq_coords'] = v_feat_unq_coords
            example['v_feat_unq_inv'] = v_feat_unq_inv
            example['v_feat_unq_cnt'] = v_feat_unq_cnt
            z_pillar_feat, occupied_mask = self.zpillar_model(example)
        if self.dsp_cfg is not None:
            features = self.dsp_model(example, group_pillar_centers, pillar_set_indices)
        else:
            group_features = self.shared_mlps(group_features)  # (1, C, L)
            group_features = group_features.transpose(1, 0).contiguous()
            pillar_features = scatter_max(group_features, pillar_set_indices, pillar_indices.shape[0])   # (C, M)
            features = pillar_features.transpose(1, 0)   # (M, C)

        if self.zpillar_cfg is not None:
            features[occupied_mask] = features[occupied_mask] + z_pillar_feat

        return spconv.SparseConvTensor(features, pillar_indices, (self.bev_height, self.bev_width), B), example

