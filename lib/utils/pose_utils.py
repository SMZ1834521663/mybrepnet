import numpy as np
import os
import math
import cv2
import torch

def pose2quaternion(poses):
    """
    Transform poses from rotation vector to quarternion
    """
    poses = poses.view(-1, 3)
    theta = torch.norm(poses, dim = 1)
    quarternion = torch.zeros([poses.shape[0], 4]).to(poses)
    quarternion[:, 0] = torch.cos(theta / 2)
    quarternion[:, 1:4] = torch.sin(theta / 2)[...,None] * poses
    return quarternion
    

def cal_attention_map(parents):
    """
    input kin
    """
    distance = np.zeros([24,24], dtype = np.float32)
    for i in range(0, 24):
        distance[i][i] = 1
    for i, j in enumerate(parents):
        distance[i][j] = 1
        distance[j][i] = 1
    d2 = np.matmul(distance, distance)
    d3 = np.matmul(d2, distance)
    d4 = np.matmul(d3, distance)
    d4[d4 != 0] = 1
    return d4

def rot_distance(rot1, rot2):
    #return 24 x N_frame x N_frame
    a = rot1.permute(1, 0, 2)
    b = rot2.permute(1, 2, 0)
    return  1 - torch.abs(a @ b)

def furthest_rotation_sampling(poses, n_samples):
    ### poses: N_frame x 24 x 4
    ### return 24 x n_samples x 4
    N_rot = poses.shape[0]
    distance = rot_distance(poses, poses)
    key_rot = []
    #for each rotation
    for i in range(0, 24):
        n_sampled = []
        candidate = [c for c in range(0, N_rot)]
        for j in range(n_samples):
            far_p = 0
            far_p_ind = 0
            for k in candidate:
                dis = -1
                for l in n_sampled:
                    if dis > distance[i][k][l] or dis == -1: 
                        dis = distance[i][k][l]
                if far_p < dis:     
                    far_p = dis
                    far_p_ind = k
            n_sampled.append(far_p_ind)
            if(far_p_ind in candidate):
                candidate.remove(far_p_ind)
        key_rot.append(poses[:,i,:][n_sampled])
    return torch.stack(key_rot, dim = 0)

def sample_rotation(cfg):
    pose_dict = []
    frame_num = cfg.num_train_frame
    for i in range(0, frame_num):
        params_path = os.path.join(cfg.train_dataset.data_root, cfg.params,
                                '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(cfg.train_dataset.data_root, cfg.params,
                    '{:06d}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        poses = params['poses'].reshape(-1, 3).astype(np.float64)
        pose_dict.append(poses)
    return pose_dict


if __name__ == "__main__":
    data_root= 'data/zju_mocap/CoreView_387'
    parents_path = os.path.join(data_root, 'lbs/parents.npy')
    parent = np.load(parents_path)
    parent[0] = 0
    attention_map = cal_attention_map(parent)
    np.save('attention_map.npy', attention_map)
    
    