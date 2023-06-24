import torch, math
from typing import List
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from . import pillar_cuda
from .group_utils import gather_feature, flatten_indices


@torch.no_grad()
def generate_pillar_indices(bev_size, point_cloud_range, point_batch_cnt, points):
    pillars, pillar_bev_indices = gen_pillar_indices(points, point_batch_cnt, bev_size, point_cloud_range)
    return pillars, pillar_bev_indices


def bev_spatial_shape(point_cloud_range, pillar_size):
    W = round((point_cloud_range[3] - point_cloud_range[0]) / pillar_size)
    H = round((point_cloud_range[4] - point_cloud_range[1]) / pillar_size)
    return int(H), int(W)


class PillarQueryAndGroup(nn.Module):
    def __init__(self, pillar_size, point_cloud_range):
        super().__init__()

        self.pillar_size = pillar_size
        self.spatial_shape = bev_spatial_shape(point_cloud_range, pillar_size)
        self.z_center = (point_cloud_range[5] + point_cloud_range[2]) / 2
        self.point_cloud_range = point_cloud_range
        self.voxel_size = [pillar_size,pillar_size,0.25]
        self.temp_size = [pillar_size,pillar_size,0.05]
        self.grid_size = torch.tensor([1440, 1440, 32]).cuda()
        self.grid_size_t = torch.tensor([1440, 1440, 10]).cuda()

    def forward(self, xyzt, xyz_batch_cnt, point_features):
        """ batch-wise operation
        Args:
            xyz: (N1+N2..., 3)  relative coordinates
            xyz_batch_cnt: (N1+N2...)
            point_features: (N1+N2..., C)
        Return:
            pillar_indices: indices for resulting sparse pillar tensor
            features: (L1+L2..., C)
        """
        # indice_pairs == -1 -> points outside range
        xyz = xyzt[:,[0,1,2]]
        xyt = xyzt[:,[0,1,4]]
        points_coords, f_center, indice_pairs = gen_indice_pairs(xyz, xyz_batch_cnt, self.pillar_size,
                                                                 self.spatial_shape, self.z_center)
        points_coords_Z, z_center, indice_pairs_z = gen_indice_pairs_3D(xyz, xyz_batch_cnt, self.voxel_size,
                                                                 self.grid_size, self.z_center)
        points_coords_T, t_center, indice_pairs_t = gen_indice_pairs_3D(xyt, xyz_batch_cnt, self.temp_size,
                                                                 self.grid_size_t, 5)
        point_set_indices, unq_inv = flatten_indices(indice_pairs) # 
        point_set_indices_z, z_unq_inv = flatten_indices(indice_pairs_z) # 
        point_set_indices_t, t_unq_inv = flatten_indices(indice_pairs_t) #         
        group_point_features = gather_feature(point_features, point_set_indices)  # (L, C)
        group_point_xyz = gather_feature(xyz, point_set_indices)  # (L, 3) [xyz]

        z_group_point_features = gather_feature(point_features, point_set_indices_z)  # (L, C)
        z_group_point_xyz = gather_feature(xyz, point_set_indices_z)  # (L, 3) [xyz]

        t_group_point_features = gather_feature(point_features, point_set_indices_t)  # (L, C)
        t_group_point_xyz = gather_feature(xyt, point_set_indices_t)  # (L, 3) [xyz]

        group_pillar_centers = gather_feature(f_center, unq_inv)  # (L, 3)  [xyz]
        group_pillar_centers = group_point_xyz - group_pillar_centers

        z_group_pillar_centers = gather_feature(z_center, z_unq_inv)  # (L, 3)  [xyz]
        z_group_pillar_centers = z_group_point_xyz - z_group_pillar_centers

        t_group_pillar_centers = gather_feature(t_center, t_unq_inv)  # (L, 3)  [xyz]
        t_group_pillar_centers = t_group_point_xyz - t_group_pillar_centers

        features = torch.cat([group_point_features.detach(), group_point_xyz.detach(),
                                    group_pillar_centers.detach()], dim=1)

        z_features = torch.cat([z_group_point_features.detach(), z_group_point_xyz.detach(),
                                    z_group_pillar_centers.detach()], dim=1)
        
        t_features = torch.cat([t_group_point_features.detach(), t_group_point_xyz.detach(),
                                    t_group_pillar_centers.detach()], dim=1)
        points_coords_Z = points_coords_Z[:,[0,3,1,2]]
        points_coords_T = points_coords_T[:,[0,3,1,2]]
        return points_coords, unq_inv, features, points_coords_Z, z_unq_inv, z_features, points_coords_T, t_unq_inv, t_features


class FGPillarQueryAndGroup(nn.Module):
    def __init__(self, pillar_size, point_cloud_range):
        super().__init__()

        self.pillar_size = pillar_size
        self.spatial_shape = bev_spatial_shape(point_cloud_range, pillar_size)
        self.z_center = (point_cloud_range[5] + point_cloud_range[2]) / 2
        self.point_cloud_range = point_cloud_range

    def forward(self, xyz, xyz_batch_cnt, point_features, bxyz):
        """ batch-wise operation
        Args:
            xyz: (N1+N2..., 3)  relative coordinates
            xyz_batch_cnt: (N1+N2...)
            point_features: (N1+N2..., C)
        Return:
            pillar_indices: indices for resulting sparse pillar tensor
            group_features: (L1+L2..., C)
        """
        pillars, pillar_centers, indice_pairs = gen_indice_pairs(xyz, xyz_batch_cnt, self.pillar_size,
                                                                 self.spatial_shape, self.z_center)

        point_set_indices, pillar_set_indices = flatten_indices(indice_pairs)
        group_point_features = gather_feature(point_features, point_set_indices)  # (L, C)
        group_point_xyz = gather_feature(xyz, point_set_indices)  # (L, 3) [xyz]
        group_point_bxyz = gather_feature(bxyz, point_set_indices)  # (L, 3) [xyz]

        group_pillar_centers = gather_feature(pillar_centers, pillar_set_indices)  # (L, 3)  [xyz]
        group_pillar_centers = group_point_xyz - group_pillar_centers
        
        group_features = torch.cat([group_point_features.detach(), group_point_xyz.detach(),
                                    group_pillar_centers.detach()], dim=1)
        
        group_point_features = torch.cat([group_point_bxyz[:,:1].detach(),
                                    group_pillar_centers.detach(), group_point_bxyz[:,1:].detach()], dim=1)
        import pdb;pdb.set_trace()
        return pillars, pillar_set_indices, group_features, group_point_features, group_point_bxyz, group_pillar_centers

class GenPillarsIndices(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, bev_size, spatial_shape):
        B = xyz_batch_cnt.numel()
        H, W = spatial_shape

        device = xyz.device
        pillar_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)

        pillar_cuda.create_pillar_indices_stack_wrapper(bev_size, xyz, xyz_batch_cnt, pillar_mask)

        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_bev_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)
        pillar_cuda.create_pillar_indices_wrapper(pillar_bev_indices, pillars)

        return pillars, pillar_bev_indices

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

gen_pillar_indices = GenPillarsIndices.apply


class GenIndicePairs(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, pillar_size, spatial_shape, z_center):
        """
        Args:
            xyz: (N1+N2..., 3+C)
            xyz_batch_cnt: (N1, N2, ...)

        Returns:
            pillars: (M1+M2..., 3) [byx]
            pillar_bev_indices: (B, H, W) none(-1)
            pillar_centers: by using pillars yx to calculate centers
            indice_pairs: (N1+N2..., K) neighboring pillars for each point
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert xyz.shape[1] == 3

        B = xyz_batch_cnt.numel()
        H, W = spatial_shape

        device = xyz.device
        pillar_mask = torch.zeros([B, H, W], dtype=torch.bool, device=device)
        pillar_cuda.create_pillar_indices_stack_wrapper(pillar_size, xyz, xyz_batch_cnt, pillar_mask)
        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_bev_indices = location.view(B, H, W) * pillar_mask - 1
        # create indices (M, 3) [byx]
        pillars = torch.zeros([M, 3], dtype=torch.int32, device=device)
        pillar_cuda.create_pillar_indices_wrapper(pillar_bev_indices, pillars)

        indice_pairs = torch.full([xyz.shape[0], 1], -1, dtype=torch.int32, device=device)

        # create pillar center [x y z]
        pillar_centers = torch.zeros([pillars.shape[0], 3], dtype=torch.float32, device=device, requires_grad=False)
        pillar_centers[:, 0] = (pillars[:, 2] + 0.5) * pillar_size
        pillar_centers[:, 1] = (pillars[:, 1] + 0.5) * pillar_size
        pillar_centers[:, 2] = z_center

        pillar_cuda.create_pillar_indice_pairs_stack_wrapper(pillar_size, xyz, xyz_batch_cnt,
                                                             pillar_bev_indices, indice_pairs)

        return pillars, pillar_centers, indice_pairs

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None, None


class GenIndicePairs_3D(Function):
    @staticmethod
    def forward(ctx, xyz:torch.Tensor, xyz_batch_cnt:torch.Tensor, pillar_size, spatial_shape, z_center):
        """
        Args:
            xyz: (N1+N2..., 3+C)
            xyz_batch_cnt: (N1, N2, ...)

        Returns:
            pillars: (M1+M2..., 3) [byx]
            pillar_bev_indices: (B, H, W) none(-1)
            pillar_centers: by using pillars yx to calculate centers
            indice_pairs: (N1+N2..., K) neighboring pillars for each point
        """
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert xyz.shape[1] == 3

        B = xyz_batch_cnt.numel()
        H, W, X = spatial_shape
        device = xyz.device
        pillar_mask = torch.zeros([B, H, W, X], dtype=torch.bool, device=device)
        #pillar_cuda.create_pillar_indices_stack_wrapper(pillar_size, xyz, xyz_batch_cnt, pillar_mask)
        pillar_size_x = pillar_size[0]
        pillar_size_z = pillar_size[2]
        pillar_cuda.create_voxel_indices_stack_wrapper(pillar_size_x, pillar_size_z, xyz, xyz_batch_cnt, pillar_mask)
        location = torch.cumsum(pillar_mask.view(-1), 0).int()
        M = location[-1].item()
        pillar_bev_indices = location.view(B, H, W, X) * pillar_mask - 1
        # create indices (M, 4) [byxz]
        pillars = torch.zeros([M, 4], dtype=torch.int32, device=device)
        pillar_cuda.create_voxel_indices_wrapper(pillar_bev_indices, pillars)

        indice_pairs = torch.full([xyz.shape[0], 1], -1, dtype=torch.int32, device=device)

        # create pillar center [x y z]
        pillar_centers = torch.zeros([pillars.shape[0], 3], dtype=torch.float32, device=device, requires_grad=False)
        pillar_centers[:, 0] = (pillars[:, 2] + 0.5) * pillar_size[0]
        pillar_centers[:, 1] = (pillars[:, 1] + 0.5) * pillar_size[0]
        pillar_centers[:, 2] = z_center

        pillar_cuda.create_voxel_indice_pairs_stack_wrapper(pillar_size_x, pillar_size_z, xyz, xyz_batch_cnt,
                                                             pillar_bev_indices, indice_pairs)

        return pillars, pillar_centers, indice_pairs

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None, None

gen_indice_pairs = GenIndicePairs.apply
gen_indice_pairs_3D = GenIndicePairs_3D.apply