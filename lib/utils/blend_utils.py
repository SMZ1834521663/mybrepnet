import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
SMPL_JOINT_IDX = {
    'pelvis_root': 0,
    'left_hip': 1,
    'right_hip': 2,
    'belly_button': 3,
    'left_knee': 4,
    'right_knee': 5,
    'lower_chest': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'upper_chest': 9,
    'left_toe': 10,
    'right_toe': 11,
    'neck': 12,
    'left_clavicle': 13,
    'right_clavicle': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_thumb': 22,
    'right_thumb': 23
}

SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}

TORSO_JOINTS_NAME = [
    'pelvis_root', 'belly_button', 'lower_chest', 'upper_chest', 'left_clavicle', 'right_clavicle'
]
TORSO_JOINTS = [
    SMPL_JOINT_IDX[joint_name] for joint_name in TORSO_JOINTS_NAME
]
BONE_STDS = np.array([0.03, 0.06, 0.03])
HEAD_STDS = np.array([0.06, 0.06, 0.06])
JOINT_STDS = np.array([0.02, 0.02, 0.02])




def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """
    wdirs: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_world_points(ppts, Rh, Th):
    """
    ppts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(ppts, Rh.transpose(1, 2)) + Th
    return pts


def pose_points_to_tpose_points(ppts, bw, A):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ppts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    pts = ppts - A[..., :3, 3]
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw, A):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * ddirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts

def pose_dir_to_can_dir(viewdir, quat):
    """transform unit viewdir from pose to canonical"""
    canview =  quaternion_to_matrix(quat).transpose(1, 2) @ viewdir.unsqueeze(-1)
    return canview.squeeze()

def transfrom_tgaussian_to_pose(pts, quat, bw, A, big_A=None):
    "transform gaussians from T to pose" 
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    if big_A is not None:
        big_A = torch.bmm(bw, big_A.view(sh[0], 24, -1))
        big_A = big_A.view(sh[0], -1, 4, 4)
        pts = pts - big_A[..., :3, 3]
        R_inv = torch.inverse(big_A[..., :3, :3])
        pts = torch.sum(R_inv * pts[:, :, None], dim=3)
        
    # if Rs is not None:
    #     A_wo_root = A[:, 1:]
    #     R_wo_root = A[..., :3, :3]
    #     R = A_wo_root @ Rs
    #     R_wo_root = torch.bmm(bw, R_wo_root.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    if big_A is not None:
        R = R @ R_inv
    r_rot = quaternion_to_matrix(quat) @ R.transpose(2, 3) 
    pts = pts + A[..., :3, 3]
    q_rot = matrix_to_quaternion(r_rot)
    return pts, q_rot, R


def transfrom_tgaussian_to_pose_snapshot(pts, quat, bw, A, Rs):
    "transform gaussians from T to pose" 
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    r_rot = Rs[0] @ quaternion_to_matrix(quat) @ R.transpose(2, 3)
    pts = (pts + A[..., :3, 3]) @ Rs[0]
    q_rot = matrix_to_quaternion(r_rot)
    return pts, q_rot 
    
    
def tpose_dirs_to_pose_dirs(ddirs, bw, A):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


def grid_sample_blend_weights(grid_coords, bw):
    # the blend weight is indexed by xyz
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]
    return bw


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x N_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, N_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], N_samples])
    y_vals = torch.rand([sh[0], N_samples])
    z_vals = torch.rand([sh[0], N_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts
