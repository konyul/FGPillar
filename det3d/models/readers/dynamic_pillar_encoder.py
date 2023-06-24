import torch
from torch import nn
from ..registry import READERS
from ..utils import build_norm_layer
from det3d.ops.pillar_ops.pillar_modules import PillarMaxPooling, FGPillarMaxPooling
import torch.nn.functional as F

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
        self.pc_range = torch.tensor(pc_range).cuda()
        self.voxel_size = torch.tensor([0.075,0.075,0.25]).cuda()
        self.point_cloud_range_t = torch.tensor([[0.01]]).cuda()
        self.temp_size = torch.tensor([[0.05]]).cuda()
        self.grid_t = torch.tensor(10).cuda()
        self.grid_size = torch.tensor([1440,1440,32]).cuda()
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
        relative[..., 4] -= 0.01
        return relative
    
    @torch.no_grad()
    def filter(self, absolute):
        points_coords_3d = torch.floor((absolute[:,:3] -  self.pc_range[0:3]) / self.voxel_size).int()
        points_coords_t = torch.round((absolute[:, [4]] - self.point_cloud_range_t) / self.temp_size).int()
        absolute_t = absolute[:,-1]
        absolute_z = absolute[:,2]
        absolute_t[points_coords_t[:,-1] > 9] = 0.45
        points_coords_z = points_coords_3d[:,-1]
        absolute_z[points_coords_z >= 32] = 2.99
        absolute_z[points_coords_z < 0] = self.pc_range[2]
        absolute[:,-1] = absolute_t
        absolute[:,2] = absolute_z
        return absolute

    def forward(self, example, **kwargs):
        if False:
            raw_points = example['points']
            coors = []
            for i, coor in enumerate(raw_points):
                coor_pad = F.pad(coor, (1, 0), 'constant', i)
                coors.append(coor_pad)
            raw_points = torch.cat(coors, dim=0)
        points_list = example.pop("points")
        device = points_list[0].device

        xyzt = []
        xyz_batch_cnt = []
        for points in points_list:
            points = self.filter(points)
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyzt.append(points[:, :5])

        xyzt = torch.cat(xyzt, dim=0).contiguous()
        pt_features = torch.cat(points_list, dim=0).contiguous()
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        sp_tensor = self.pfn_layers(xyzt, xyz_batch_cnt, pt_features)
        return sp_tensor


@READERS.register_module
class DynamicFGPillarFeatureNet(nn.Module):
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
        self.pfn_layers = FGPillarMaxPooling(
            mlps=num_filters,
            pillar_size=pillar_size,
            point_cloud_range=pc_range,
            dsp_cfg=kwargs.get("dsp_cfg", None),
            zpillar_cfg=kwargs.get("zpillar_cfg",None)
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
        xyz = []
        xyz_batch_cnt = []
        bxyz = []
        for idx, points in enumerate(points_list):
            coor_pad = F.pad(points, (1, 0), mode='constant', value=idx)
            bxyz.append(coor_pad)
            
            points = self.absl_to_relative(points)

            xyz_batch_cnt.append(len(points))
            xyz.append(points[:, :3])
            
        bxyz = torch.cat(bxyz, axis=0).contiguous()
        xyz = torch.cat(xyz, dim=0).contiguous()
        pt_features = torch.cat(points_list, dim=0).contiguous()
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, dtype=torch.int32).to(device)

        sp_tensor, example = self.pfn_layers(xyz, xyz_batch_cnt, pt_features, bxyz, example)
        return sp_tensor, example