from lib.config import cfg
from lib.train import Recorder 
from lib.train.trainers.gaussian_human import GaussianHuman
from lib.datasets.make_dataset import make_data_loader, make_dataset
from lib.datasets.monocular_novel_view import Dataset
from lib.utils.vis_3d import *
from lib.datasets.load_data import *
import torch.multiprocessing
import torch
import sys
import os
import glob

#python animate.py mod=test test_novel_pose=True subject=zju377
def load_pose(data_root):
    params_path_root = os.path.join(data_root, cfg.params)
    param_list = sorted(glob.glob(params_path_root + "/*.npy"))
    poses = [] 
    for param_path in param_list:
        params = np.load(param_path, allow_pickle=True).item()
        pose = params['poses'].reshape(-1, 3)
        poses.append(pose)
    np.stack(poses, axis=0)
    return poses#, nearest_frame_index, canonical_joints 

print('loading animation data')
dataset = make_dataset(cfg, cfg.mod)
params_path_root = '/home/wzx2021/gaussian-human/data/zju_mocap/CoreView_393'
poses = load_pose(params_path_root)
test_data = load_data(cfg.data_name, dataset)
gaussian_human = GaussianHuman(cfg, test_data)
ckpt_dir = os.path.join(cfg.trained_model_dir + "/chkpnt" + "latest.pth")  
recorder = Recorder(cfg)
if not os.path.exists(ckpt_dir):
    print("checkpoint not found!")
else: 
    model_params, first_iter = torch.load(ckpt_dir)
    gaussian_human.load(model_params, cfg)
    recorder_scalar = {'step': first_iter}
    recorder.load_state_dict(recorder_scalar)
gaussian_human.animate(test_data['data'], poses, recorder)


