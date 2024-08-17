import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.networks import embedder
from lib.utils.pose_utils import pose2quaternion
from lib.utils.vis_3d import *
from lib.utils import sample_utils
from lib.utils.blend_utils import transfrom_tgaussian_to_pose 
from lib.networks.volume import CanoBlendWeightVolume
from lib.utils.gaussian_utils import * 
import tinycudann as tcnn
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.networks.mlp_delta_body_pose import BodyPoseRefiner, RodriguesModule
import os

def back_forward_fn(grad):
    print(grad)
    return grad

def rodrigue(rvec):
        r''' Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        '''
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), 
        dim=1).view(-1, 3, 3)
        
def get_rigid_transformation(poses, joints, parents, correction=None):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = rodrigue(poses.view(-1, 3))
    
    if correction is not None:
        rot_wo_root = rot_mats[1:]
        rot_wo_root = rot_wo_root @ correction
        rot_mats = torch.cat([rot_mats[:1], rot_wo_root], dim = 0)
    
    # obtain the relative joints
    rel_joints = joints.clone()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = torch.cat([rot_mats, rel_joints[..., None]], axis=2)
    padding = torch.zeros([24, 1, 4], device=poses.device)
    padding[..., 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=0)

    posed_joints = transforms[:, :3, 3].clone()

    # obtain the rigid transformation
    padding = torch.zeros([24, 1], device = poses.device)
    joints_homogen = torch.cat([joints, padding], axis=1)
    rel_joints = torch.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    # transforms = transforms.astype(np.float32)
    # if return_joints:
        # return transforms, posed_joints
    # else:
    return transforms

def normalize(pts, bbox):
    c = (bbox[0] + bbox[1]) / 2
    s = (bbox[1] - bbox[0])
    center = c
    scale = s
    bbox = bbox
    pts = (pts - center) / scale + 0.5
    return pts


class DeformNetwork_pose(nn.Module):
    def __init__(self, cfg):
        super(DeformNetwork_pose, self).__init__()
        self.xyz_encoder = tcnn.Encoding(n_input_dims=3, 
                                            encoding_config={
                                             "otype": "HashGrid",
                                            "n_levels": 16,
                                            "n_features_per_level": 2,
                                            "log2_hashmap_size": 19,
                                            "base_resolution": 16,
                                            "per_level_scale": 1.5
                                            })
        input_ch = 92 + (self.xyz_encoder.n_output_dims if cfg.hash_grid else embedder.xyz_dim)
        D = 2
        W = 128
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.cfg = cfg
        self.weight_volume = CanoBlendWeightVolume(os.path.join(self.cfg.data_root, 'lbs', 'cano_weight_volume.npz'))
        self.resd_fc0 = nn.Conv1d(W, W, 1)
        self.resd_fc1 = nn.Conv1d(W, 3, 1)
        self.res_quat_fc0 = nn.Conv1d(W, W, 1)
        self.res_quat_fc1 = nn.Conv1d(W, 4, 1)
        self.res_scale_fc0 = nn.Conv1d(W, W, 1)
        self.res_scale_fc1 = nn.Conv1d(W, 3, 1)
        
        self.actvn = nn.ReLU()
        init_val = 1e-5
        self.resd_fc1.weight.data.uniform_(-init_val, init_val)
        self.res_quat_fc1.weight.data.uniform_(-init_val, init_val)
        self.res_scale_fc1.weight.data.uniform_(-init_val, init_val)
        self.resd_fc1.bias.data.fill_(0)
        self.res_quat_fc1.bias.data.fill_(0)
        self.res_scale_fc1.bias.data.fill_(0)


    def calculate_residual_deformation(self, tpose, quat_pose, scale, tbounds):
        # tpose.register_hook(back_forward_fn)
        tpose = normalize(tpose, tbounds)
        if self.cfg.hash_grid: 
            pts = self.xyz_encoder(tpose[0])[None]
        else: 
            pts = embedder.xyz_embedder(tpose)
        pts = pts.transpose(1, 2)
        # pts.register_hook(back_forward_fn)
        latent = quat_pose.view(1, 92)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        # net as point concat pose feature
        # net = torch.cat((features, net), dim=1)
        resd = self.resd_fc1(self.actvn(self.resd_fc0(net)))
        res_quat = self.res_quat_fc1(self.actvn(self.res_quat_fc0(net)))
        res_quat = res_quat.transpose(1, 2)
        # res_quat = 0.05 * torch.tanh(res_quat) #changed
        res_scale = torch.zeros_like(scale)
        if(self.cfg.scalable):
            res_scale = self.res_scale_fc1(self.actvn(self.res_scale_fc0(net)))
            res_scale = res_scale.transpose(1, 2)
            # res_scale = 0.05 * torch.tanh(res_scale) #changed
        resd = resd.transpose(1, 2)
        resd = 0.05 * torch.tanh(resd)
        return resd, res_quat, res_scale, net


    def forward(self, view, xyz, rot, scale, smpl_utils, iter, Rs=None):
        "rot: quaternion representation"
        #TODO change to pytorch3d
        poses = view['poses']
        joints = smpl_utils['joints']
        parents = smpl_utils['parents']
        quat_poses = pose2quaternion(poses[3:])
        if self.cfg.pose_correction:
            A = get_rigid_transformation(poses, joints, parents, correction = Rs[0])
        else:
            A = get_rigid_transformation(poses, joints, parents)

        with torch.no_grad():
            if self.cfg.volume_weight:
                pbw = self.weight_volume.forward_weight(xyz, requires_scale=True)
            else:    
                pbw, _ = sample_utils.sample_blend_closest_points(xyz[None], smpl_utils['tvertices'][None], smpl_utils['weights'], exp=1e-5)
        pbw = pbw.permute(0, 2, 1)
        resd = torch.zeros_like(xyz[None])
        res_quat = torch.zeros_like(rot[None])
        res_scale = torch.zeros_like(scale[None])
        # xyz = self.normalize(xyz)
        resd, res_quat, res_scale, pose_latent = self.calculate_residual_deformation(xyz[None], quat_poses, scale, smpl_utils['tbounds'])
        if self.cfg.use_big_pose:
            big_A = smpl_utils['big_A']
        if self.cfg.can_res:
            xyz = xyz + resd
            rot =  torch.nn.functional.normalize(rot + res_quat, dim=-1)
            init_ppose, init_quat, R = transfrom_tgaussian_to_pose(xyz, rot, pbw, A, big_A)
            scale = scale + res_scale
            return init_ppose, resd, init_quat, res_quat, scale, res_scale, pose_latent, R
        init_ppose, init_quat = transfrom_tgaussian_to_pose(xyz[None], rot[None], pbw, A, big_A)
        if self.cfg.resd:
            xyz = init_ppose + resd
        else:
            xyz = init_ppose
        if self.cfg.res_quat:
            quat =  torch.nn.functional.normalize(init_quat + res_quat)
        else:
            quat = init_quat 
        scale = scale + res_scale
        return xyz, resd, quat, res_quat, scale, res_scale, pose_latent, R
