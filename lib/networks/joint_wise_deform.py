import torch.nn as nn
from lib.networks import embedder
from lib.utils.pose_utils import pose2quaternion
from lib.utils.vis_3d import *
from lib.utils import sample_utils
from lib.utils.blend_utils import transfrom_tgaussian_to_pose 
from lib.utils. import * 

def back_forward_fn(grad):
    print(grad)
    return grad
        
class LocalDeformNetwork(nn.Module):
    def __init__(self):
        super(LocalDeformNetwork, self).__init__()
        input_ch = 4 + embedder.xyz_dim
        D = 4
        W = 256
        self.skips = [4]
        self.module_list = nn.ModuleList()
        self.resd_fc = nn.ModuleList()
        self.res_scale_fc = nn.ModuleList()
        self.res_quat_fc = nn.ModuleList()
        for j in range(24):
            module = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
            resd_fc = nn.Conv1d(W, 3, 1)
            res_quat_fc = nn.Conv1d(W, 4, 1)
            res_scale_fc = nn.Conv1d(W, 3, 1)
            resd_fc.bias.data.fill_(0)
            self.module_list.append(module)
            self.resd_fc.append(resd_fc)
            self.res_scale_fc.append(res_scale_fc)
            self.res_quat_fc.append(res_quat_fc)
        self.actvn = nn.ReLU()

    def calculate_residual_deformation(self, cfg, local_xyz, quat_pose, scale, j):
        pts = embedder.xyz_embedder(local_xyz).float()
        pts = pts.transpose(1, 2)
        # pts.register_hook(back_forward_fn)
        latent = quat_pose.transpose(1, 2)

        # B X (embed_dim + 4) X N
        features = torch.cat((pts, latent), dim=1)

        net = features
        for i, l in enumerate(self.module_list[j]):
            net = self.actvn(self.module_list[j][i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc[j](net)
        res_quat = self.res_quat_fc[j](net)
        res_quat = res_quat.transpose(1, 2)
        res_scale = torch.ones_like(scale)
        if(cfg.scalable):
            res_scale = 0.05 * torch.tanh(self.res_scale_fc[j](net).transpose(1, 2)) + 1
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd, res_quat, res_scale

    #TODO precompute skinning
    def forward(self, cfg, batch, xyz, rot, scale=None):
        "rot: quaternion representation"
        joints = batch['joints'][0]
        quat_poses = pose2quaternion(batch['poses'])

        with torch.no_grad():
            pbw, _ = sample_utils.sample_blend_closest_points(xyz[None], batch['tvertices'], batch['weights'], exp=1e-5)
            pbw = pbw.permute(0, 2, 1)

        init_ppose, init_quat = transfrom_tgaussian_to_pose(xyz[None], rot[None], pbw, batch['A'])
        local_xyz = xyz[:, None, :] - joints
        quat_poses = quat_poses[None].repeat(local_xyz.shape[0], 1, 1)

        local_xyz = local_xyz.permute(1, 0, 2)
        quat_poses = quat_poses.permute(1, 0, 2)
        # J x N x 7
        # local_signal = torch.cat([local_xyz, quat_poses], dim = -1).permute(1, 0, 2)

        resd = []
        res_quat = []
        delta_scale = []
        for j in range(len(joints)):
            resd_, res_quat_, delta_scale_ = self.calculate_residual_deformation(cfg, local_xyz[j][None], quat_poses[j][None], scale, j)
            resd.append(resd_)
            res_quat.append(res_quat_)
            delta_scale.append(delta_scale_)

        resd = torch.cat(resd, dim=0).transpose(0, 1)
        res_quat = torch.cat(res_quat, dim=0).transpose(0, 1)
        delta_scale = torch.cat(delta_scale, dim=0).transpose(0, 1)
        ind = pbw[0].argmax(0).squeeze()
        resd = torch.gather(resd, 1, ind.view(-1, 1, 1).expand(-1, 1, 3)).squeeze()
        res_quat = torch.gather(res_quat, 1, ind.view(-1, 1, 1).expand(-1, 1, 4)).squeeze()
        delta_scale = torch.gather(delta_scale, 1, ind.view(-1, 1, 1).expand(-1, 1, 3)).squeeze()
      

        if cfg.resd:
            xyz = init_ppose + resd
        else:
            xyz = init_ppose
        if cfg.res_quat:
            quat =  torch.nn.functional.normalize(init_quat + res_quat)
        else:
            quat = init_quat 
        if cfg.scalable:
            scale = scale * delta_scale
        return xyz, resd, quat, res_quat, scale, delta_scale







    