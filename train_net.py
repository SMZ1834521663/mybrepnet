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

MODULE_PATH = os.path.abspath(".")
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(cfg):
    print('loading data')
    dataset = make_dataset(cfg, cfg.mod)
    train_data = load_data(cfg.data_name, dataset)
    recorder = Recorder(cfg)
    gaussian_human = GaussianHuman(cfg, train_data)
    first_iter = 1
    if(cfg.resume):
        ckpt_dir = os.path.join(cfg.trained_model_dir + "/chkpnt" + "latest.pth")  
        if not os.path.exists(ckpt_dir):
            print("checkpoint not found!")
        else: 
            model_params, first_iter = torch.load(ckpt_dir)
            gaussian_human.load(model_params, cfg)
            recorder_scalar = {'step': first_iter}
            recorder.load_state_dict(recorder_scalar)
    debug_img_dir = f'log/{cfg.exp_name}'
    if(not os.path.exists(debug_img_dir)):
        os.makedirs(debug_img_dir)
    gaussian_human.train(first_iter, train_data['data'], recorder)

def test(cfg):
    print('loading test data')
    dataset = make_dataset(cfg, cfg.mod)
    print(len(dataset))
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
    gaussian_human.val(test_data['data'], recorder)

def test_ckpt(cfg):
    dataset = make_dataset(cfg, cfg.mod)
    test_data = load_data(cfg.data_name, dataset)
    gaussian_human = GaussianHuman(cfg, test_data)
    for i in range(3000, 15000, 2000):
        ckpt_dir = os.path.join(cfg.trained_model_dir + "/chkpnt" + str(i) + ".pth")  
        recorder = Recorder(cfg)
        if not os.path.exists(ckpt_dir):
            print("checkpoint not found!")
        else: 
            model_params, first_iter = torch.load(ckpt_dir)
            gaussian_human.load(model_params, cfg)
            recorder_scalar = {'step': first_iter}
            recorder.load_state_dict(recorder_scalar)
        print(i)
        gaussian_human.val(test_data['data'], recorder)


def vis_novel_view(cfg):
    from torch.utils.data import DataLoader
    num_frames = 60
    gaussian_human = GaussianHuman(cfg)
    dataset = Dataset(num_frames, cfg.data_root, cfg.human, cfg.ann_file, cfg.mod)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    print(f'free view dataloader {len(dataloader)}')
    ckpt_dir = os.path.join(cfg.trained_model_dir + "/chkpnt" + "latest.pth")  
    if not os.path.exists(ckpt_dir):
        print("checkpoint not found!")
    else: 
        model_params, first_iter = torch.load(ckpt_dir)
        gaussian_human.load(model_params, cfg)
    gaussian_human.free_view(dataloader)

if __name__ == "__main__":
    if cfg.vis_free_view:
        cfg.mod = 'test'
        vis_novel_view(cfg)
    elif(cfg.mod == 'train'):
        train(cfg)
    elif(cfg.mod == 'test'):
        if cfg.check_ckpt:
            test_ckpt(cfg) 
        else:
            test(cfg)
