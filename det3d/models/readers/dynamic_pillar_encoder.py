import torch
from torch import nn
from ..registry import READERS
from ..utils import build_norm_layer
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling, FGPillarMaxPooling
import torch.nn.functional as F
import torch_scatter
from det3d.models.utils.mlp import build_mlp
try:
    import spconv.pytorch as spconv
except:
    import spconv
@READERS.register_module
class DynamicPillarFeatureNet(nn.Module):
    def __init__(
        self,
        num_input_features=2,
        num_filters=(32,),
        pillar_size=0.1,
        virtual=False,
        pc_range=(0, -40, -3, 70.4, 40, 1),
        **kwargs
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param pillar_size: (<float>: 3). Size of pillars.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.pc_range = pc_range
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_filters = [6 + num_input_features] + list(num_filters)
        self.pfn_layers = PillarMaxPooling(
            mlps=num_filters,
            pillar_size=pillar_size,
            point_cloud_range=pc_range
        )

        self.virtual = virtual

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    def forward(self, example, **kwargs):
        points_list = example.pop("points")
        device = points_list[0].device

        if self.virtual:
            # virtual_point_mask = features[..., -2] == -1
            # virtual_points = features[virtual_point_mask]
            # virtual_points[..., -2] = 1
            # features[..., -2] = 0
            # features[virtual_point_mask] = virtual_points
            raise NotImplementedError

        xyz = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])

        xyz = torch.cat(xyz, dim=0).contiguous()
        pt_features = torch.cat(points_list, dim=0).contiguous()
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        sp_tensor = self.pfn_layers(xyz, xyz_batch_cnt, pt_features)
        return sp_tensor

class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated
@READERS.register_module
class DynamicPillarFeatureNet_HBAM(nn.Module):
    def __init__(
        self,
        num_input_features=2,
        num_filters=(32,),
        pillar_size=0.1,
        virtual=False,
        pc_range=(0, -40, -3, 70.4, 40, 1),
        **kwargs
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param pillar_size: (<float>: 3). Size of pillars.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """
        super().__init__()
        self.point_cloud_range =  point_cloud_range = torch.tensor(pc_range).cuda()
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_filters = [6 + num_input_features] + list(num_filters)

        self.virtual = virtual
        self.voxel_size = voxel_size = torch.tensor([0.075, 0.075, 0.25]).cuda()
        self.temp_size = torch.tensor([0.05]).cuda()
        self.point_cloud_range_t = torch.tensor([0.01]).cuda()
        self.grid_size = grid_size = torch.tensor([1440,1440,32]).cuda()
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        self.scale_z = grid_size[2]
        self.grid_t = torch.tensor(10).cuda()
        self.scale_xyt = grid_size[0] * grid_size[1] * self.grid_t
        self.scale_yt = grid_size[1] * self.grid_t
        self.scale_t = self.grid_t
        self.zpillar_model = build_mlp()
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = -1
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, True, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.bev_height = int(self.grid_size[0])
        self.bev_width = int(self.grid_size[1])

    def dyn_voxelization(self, points, point_coords, batch_dict):
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
        batch_dict['v_unq_inv'] = unq_inv
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict
    
    def dyn_voxelization_t(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyt + \
                        point_coords[:, 0] * self.scale_yt + \
                        point_coords[:, 1] * self.scale_t + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyt,
                                    (unq_coords % self.scale_xyt) // self.scale_yt,
                                    (unq_coords % self.scale_yt) // self.scale_t,
                                    unq_coords % self.scale_t), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['t_unq_inv'] = unq_inv
        batch_dict['toxel_features'] = points_mean.contiguous()
        batch_dict['toxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict
    def forward(self, example, **kwargs):
        raw_points = example['points']
        coors = []
        for batch, coor in enumerate(raw_points):
            coor_pad = F.pad(coor, (1, 0), 'constant', batch)
            coors.append(coor_pad)
        points = torch.cat(coors, dim=0)
        

        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        points_coords = points_coords_3d[:,:2]
        points_coords_t = torch.round((points[:, [5]] - self.point_cloud_range_t) / self.temp_size).int()
        points_coords_t[points_coords_t[:,-1]>9] = 9
        points_coords_t = torch.cat([points_coords, points_coords_t],dim=-1)
        
        points_coords_z = points_coords_3d[:,2:]
        points_coords_z[points_coords_z>=32] =31
        points_coords_z[points_coords_z<0] =0
        points_coords_3d = torch.cat([points_coords, points_coords_z],dim=-1)
        mask = ((points_coords_3d >= 0) & (points_coords_3d < self.grid_size)).all(dim=1)
        points_coords_3d = points_coords_3d[mask]
        points_coords_t = points_coords_t[mask]
        points = points[mask]
        points_coords = points_coords[mask]

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        points_xyz = points[:, [1, 2, 3]].contiguous()
        batch_dict = {}
        device = points_xyz.device
        pillar_centers = torch.zeros([points_xyz.shape[0], 3], dtype=torch.float32, device=device, requires_grad=False)

        xyz = points_xyz - self.point_cloud_range[:3]
        pillar_centers[:, 0] = (points_coords[:, 0] + 0.5) * 0.075
        pillar_centers[:, 1] = (points_coords[:, 1] + 0.5) * 0.075
        pillar_centers[:, 2] = self.z_offset
        
        pillar_centers = xyz - pillar_centers
        
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]
        
        pt_features = [points, xyz, pillar_centers]
        pt_features = torch.cat(pt_features, dim=-1)
        batch_dict = self.dyn_voxelization(points, points_coords_3d, batch_dict)
        batch_dict = self.dyn_voxelization_t(points, points_coords_t, batch_dict)
        batch_dict['pillar_merge_coords'] = merge_coords
        batch_dict['unq_inv'] = unq_inv
        batch_dict['points'] = points
        batch_dict['point_cloud_range'] = self.point_cloud_range
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['grid_size'] = self.grid_size
        voxel_features, voxel_features_coords = batch_dict['voxel_features'], batch_dict['voxel_features_coords']
        v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:, 3] * self.scale_y + voxel_features_coords[:, 2]
        v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        batch_dict['v_feat_unq_coords'] = v_feat_unq_coords
        batch_dict['v_feat_unq_inv'] = v_feat_unq_inv
        batch_dict['voxel_features'] = voxel_features
        batch_dict['v_feat_unq_cnt'] = v_feat_unq_cnt
        toxel_features, toxel_features_coords = batch_dict['toxel_features'], batch_dict['toxel_features_coords']
        t_feat_coords = toxel_features_coords[:, 0] * self.scale_xy + toxel_features_coords[:, 3] * self.scale_y + toxel_features_coords[:, 2]
        t_feat_unq_coords, t_feat_unq_inv, t_feat_unq_cnt = torch.unique(t_feat_coords, return_inverse=True, return_counts=True, dim=0)
        batch_dict['t_feat_unq_coords'] = t_feat_unq_coords
        batch_dict['t_feat_unq_inv'] = t_feat_unq_inv
        batch_dict['toxel_features'] = toxel_features
        batch_dict['t_feat_unq_cnt'] = t_feat_unq_cnt
        z_pillar_feat, occupied_mask = self.zpillar_model(batch_dict)
            
        if False:
            if self.use_cluster_xyz:
                features = self.dsp_model(batch_dict, f_center, unq_inv, f_cluster)
            else:
                features = self.dsp_model(batch_dict, f_center, unq_inv)
        else:
            pt_features = pt_features[:,1:].contiguous()
            for pfn in self.pfn_layers:
                features = pfn(pt_features, unq_inv)

        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]
        features[occupied_mask] = features[occupied_mask] + z_pillar_feat
        sp_tensor = spconv.SparseConvTensor(features, pillar_coords, (self.bev_height, self.bev_width), batch+1)
        return sp_tensor