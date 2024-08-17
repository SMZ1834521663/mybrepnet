import sys
sys.path.append('/home/wzx2021/gaussian-human/')
import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.vis_3d import *
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getSmpl2view
from lib.utils.sample_utils import random_points_on_meshes_with_face_and_bary
from pytorch3d.structures import Meshes
from lib.utils.body_util import approx_gaussian_bone_volumes
from lib.utils.blend_utils import tpose_points_to_pose_points
import glob

def merge_params(data_root):
    params_root = os.path.join(data_root, 'params', '*.npy')
    Rh = []
    Th = []
    poses = []
    for i in glob.glob(params_root):
        params = np.load(i, allow_pickle=True).item()
        Rh.append(params['Rh'].astype(np.float32))
        Th.append(params['Th'].astype(np.float32))
        poses.append(params['poses'].reshape(-1, 3))
    Rh = np.stack(Rh)
    Th = np.stack(Th)
    poses = np.stack(poses)
    merged_params={
        "Rh": Rh,
        "Th": Th,
        "poses": poses
    }    
    np.save(os.path.join(data_root, "merged_params.npy"), merged_params)

        

if __name__ == "__main__":
    merge_params("/home/wzx2021/gaussian-human/data/zju_mocap/CoreView_377")
    