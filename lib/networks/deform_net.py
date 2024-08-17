import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.networks import embedder
from lib.utils.pose_utils import pose2quaternion
from lib.utils.vis_3d import *
from lib.utils import sample_utils
from lib.utils.blend_utils import transfrom_tgaussian_to_pose, transfrom_tgaussian_to_pose_snapshot 
from lib.networks.volume import CanoBlendWeightVolume
from lib.utils.gaussian_utils import * 
import tinycudann as tcnn
import os

def back_forward_fn(grad):
    print(grad)
    return grad
        
class DeformNetwork(nn.Module):
    def __init__(self, cfg):
        super(DeformNetwork, self).__init__()
        input_ch = 159
        D = 4
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.cfg = cfg
        self.weight_volume = CanoBlendWeightVolume(os.path.join(self.cfg.data_root, 'lbs', 'cano_weight_volume.npz'))
        print(self.cfg.data_root)
        self.resd_fc0 = nn.Conv1d(W + input_ch, W, 1)
        self.resd_fc1 = nn.Conv1d(W, 3, 1)
        self.res_quat_fc0 = nn.Conv1d(W + input_ch, W, 1)
        self.res_quat_fc1 = nn.Conv1d(W, 4, 1)
        self.res_scale_fc0 = nn.Conv1d(W + input_ch, W, 1)
        self.res_scale_fc1 = nn.Conv1d(W, 3, 1)
        self.actvn = nn.ReLU()
        if cfg.init_resq:
            init_val = 1e-5
            self.resd_fc1.weight.data.uniform_(-init_val, init_val)
            self.res_quat_fc1.weight.data.uniform_(-init_val, init_val)
            self.res_scale_fc1.weight.data.uniform_(-init_val, init_val)
        self.resd_fc1.bias.data.fill_(0)
        self.res_quat_fc1.bias.data.fill_(0)
        self.res_scale_fc1.bias.data.fill_(0)


    def calculate_residual_deformation(self, tpose, quat_pose, scale):
        # tpose.register_hook(back_forward_fn)
        pts = embedder.xyz_embedder(tpose)
        pts = pts.transpose(1, 2)
        # pts.register_hook(back_forward_fn)
        latent = quat_pose.view(1, 96)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)

        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        net = torch.cat((features, net), dim=1)
        resd = self.resd_fc1(self.actvn(self.resd_fc0(net)))
        res_quat = self.res_quat_fc1(self.actvn(self.res_quat_fc0(net)))
        res_quat = res_quat.transpose(1, 2)
        res_quat = 0.05 * torch.tanh(res_quat)
        res_scale = torch.zeros_like(scale)
        if(self.cfg.scalable):
            res_scale = self.res_scale_fc1(self.actvn(self.res_scale_fc0(net)))
            res_scale = res_scale.transpose(1, 2)
            res_scale = 0.05 * torch.tanh(res_scale)
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd, res_quat, res_scale


    def forward(self, batch, xyz, rot, scale, iter):
        "rot: quaternion representation"
        #TODO change to pytorch3d
        quat_poses = pose2quaternion(batch['poses'])
        with torch.no_grad():
            if self.cfg.volume_weight:
                pbw = self.weight_volume.forward_weight(xyz, requires_scale=True)
            else:    
                pbw, _ = sample_utils.sample_blend_closest_points(xyz[None], batch['tvertices'], batch['weights'], exp=1e-5)
        pbw = pbw.permute(0, 2, 1)
        resd = torch.zeros_like(xyz[None])
        res_quat = torch.zeros_like(rot[None])
        res_scale = torch.zeros_like(scale[None])
        resd, res_quat, res_scale = self.calculate_residual_deformation(xyz[None], quat_poses, scale)
    
        if self.cfg.can_res:
            xyz = xyz + resd
            rot =  torch.nn.functional.normalize(rot + res_quat, dim=-1)
            if self.cfg.snapshot:
                # if self.cfg.vis_free_view:
                # Rs = batch['R_gt']
                Rs = batch['Rs']
                # else:
                    # Rs = batch['Rs']
                init_ppose, init_quat = transfrom_tgaussian_to_pose_snapshot(xyz, rot, pbw, batch['A'], Rs)
            else:
                init_ppose, init_quat = transfrom_tgaussian_to_pose(xyz, rot, pbw, batch['A'])

            scale = scale + res_scale
            return init_ppose, resd, init_quat, res_quat, scale, res_scale 
        init_ppose, init_quat = transfrom_tgaussian_to_pose(xyz[None], rot, pbw, batch['A'])
        if self.cfg.resd:
            xyz = init_ppose + resd
        else:
            xyz = init_ppose
        if self.cfg.res_quat:
            quat = torch.nn.functional.normalize(init_quat + res_quat)
        else:
            quat = init_quat 
        scale = scale + res_scale
        return xyz, resd, quat, res_quat, scale, res_scale 
