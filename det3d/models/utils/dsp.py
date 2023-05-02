import torch
from torch import nn
import torch_scatter
from det3d.utils.fuse import fold_conv_bn_eval_sequential
class DSP(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        num_point_features = model_cfg.num_point_features
        intermediate_feature = model_cfg.get("intermediate_feature", 32)
        self.AVFE_point_feature_fc = nn.Sequential(nn.Linear(num_point_features, intermediate_feature, bias=False),
                                                        nn.BatchNorm1d(intermediate_feature, eps=1e-3, momentum=0.01),
                                                        nn.ReLU())
        self.use_shift = model_cfg.get("use_shift", False)
        self.use_downsample = model_cfg.get("use_downsample", False)
        self.use_upsample = model_cfg.get("use_upsample", False)
        self.point_cloud_range = torch.tensor(model_cfg.get("point_cloud_range")).cuda()
        self.voxel_size = torch.tensor(model_cfg.get("voxel_size")).cuda()
        self.grid_size = torch.tensor(model_cfg.get("grid_size")).cuda()
        self.x_offset = model_cfg.get("x_offset")
        self.y_offset = model_cfg.get("y_offset")
        self.z_offset = model_cfg.get("z_offset")
        self.voxel_x = model_cfg.get("voxel_x")
        self.voxel_y = model_cfg.get("voxel_y")
        self.bn_folding = model_cfg.get("bn_folding")
        
        in_channel = intermediate_feature * 2
        if self.use_shift:
            in_channel += intermediate_feature * 2
        if self.use_downsample:
            in_channel += intermediate_feature * 2
        if self.use_upsample:
            in_channel += intermediate_feature * 2
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                    nn.Linear(in_channel, 32, bias=False),
                                                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                                                    nn.ReLU())
        
    def scale(self, points, downsample_level, points_coords_3d):
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        points_z = points_coords_3d[:,-1:]
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0,1]]) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                       points_coords[:, 0] * scale_y + \
                       points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset


        features = self.gen_feat(points, f_center, unq_inv)
        return features
    
    def shift(self, points, points_coords_3d):
        shifted_point_cloud_range = self.point_cloud_range[[0,1]] + self.voxel_size[[0,1]] / 2
        points_coords = (torch.floor((points[:, [1, 2]] - shifted_point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]) + 1).int()
        points_z = points_coords_3d[:,-1:]
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        shifted_scale_xy = (self.grid_size[0] + 1) * (self.grid_size[1] + 1)
        shifted_scale_y = (self.grid_size[1] + 1)
        merge_coords = points[:, 0].int() * shifted_scale_xy + \
                       points_coords[:, 0] * shifted_scale_y + \
                       points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        
        features = self.gen_feat(points, f_center, unq_inv)
        return features
    def to_dense_batch(self, x, pillar_idx, max_points, max_pillar_idx):
        r"""
        Point sampling according to pillar index with constraint amount
        """
        
        # num_points in pillars (0 for empty pillar)
        num_nodes = torch_scatter.scatter_add(pillar_idx.new_ones(x.size(0)), pillar_idx, dim=0,
                                dim_size=max_pillar_idx)
        cum_nodes = torch.cat([pillar_idx.new_zeros(1), num_nodes.cumsum(dim=0)])

        # check if num_points in pillars exceed the predefined num_points value
        filter_nodes = False
        if num_nodes.max() > max_points:
            filter_nodes = True
        tmp = torch.arange(pillar_idx.size(0), device=x.device) - cum_nodes[pillar_idx]
        if filter_nodes:
            mask = tmp < max_points
            x = x[mask]
            pillar_idx = pillar_idx[mask]
        return x, pillar_idx
    
    def gen_feat(self, points, f_center, unq_inv):
        features = [f_center, points[:,1:]]
        features = torch.cat(features,dim=-1).contiguous()
        if self.bn_folding:
            if not self.AVFE_point_feature_fc.training:
                fused_AVFE_point_feature_fc = fold_conv_bn_eval_sequential(self.AVFE_point_feature_fc)
                scatter_feature = fused_AVFE_point_feature_fc(features)
            else:
                scatter_feature = self.AVFE_point_feature_fc(features) 
        else:
            scatter_feature = self.AVFE_point_feature_fc(features) 
        x_mean = torch_scatter.scatter_mean(scatter_feature, unq_inv, dim=0)
        features = torch.cat([scatter_feature, x_mean[unq_inv, :]], dim=1)
        return features

    def forward(self, batch_dict, f_center, unq_inv):
        points = batch_dict['points']
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()

        if self.use_shift:
                shifted_features = self.shift(points, points_coords_3d)
        if self.use_downsample:
            downsampled_features = self.scale(points, 2, points_coords_3d)
        if self.use_upsample:
            upsampled_features = self.scale(points, 1/2, points_coords_3d)
        features = self.gen_feat(points, f_center, unq_inv)
        final_features = [features]
        if self.use_shift:
            final_features.append(shifted_features)
        if self.use_downsample:
            final_features.append(downsampled_features)
        if self.use_upsample:
            final_features.append(upsampled_features)
        final_features = torch.cat(final_features, dim=-1).contiguous()
        if self.bn_folding:
            if not self.AVFEO_point_feature_fc.training:
                fused_AVFEO_point_feature_fc = fold_conv_bn_eval_sequential(self.AVFEO_point_feature_fc)
                final_features_fc = fused_AVFEO_point_feature_fc(final_features)
            else:
                final_features_fc = self.AVFEO_point_feature_fc(final_features)
        else:
            final_features_fc = self.AVFEO_point_feature_fc(final_features)
        batch_dict['dsp_feat'] = final_features_fc
        features = torch_scatter.scatter_max(final_features_fc, unq_inv, dim=0)[0]
        return features
def build_dsp(model_cfg, model_name='DSP'):
    model_dict = {
        'DSP': DSP
}
    model_class = model_dict[model_name]

    model = model_class(model_cfg
                        )
    return model
