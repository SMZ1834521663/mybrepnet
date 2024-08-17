from .transforms import make_transforms
from . import samplers
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
if cfg.snapshot:
    from lib.datasets.monocular_gaussian_dataset import Dataset
else:
    from lib.datasets.gaussian_dataset_acc_merge import Dataset
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')

def make_dataset(cfg, mod):
    print(f"Initailizing {mod}ing dataset")
    dataset = Dataset(cfg.data_root, cfg.human, cfg.ann_file, mod)
    return dataset

def make_data_sampler(dataset, shuffle, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler_thu':
        sampler = samplers.FrameSampler_thu(dataset)
        return sampler
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_eval = False, max_iter=-1):
    if cfg.mod == 'train' and not is_eval:
        batch_size = cfg.train.batch_size
        shuffle = True 
        shuffle = cfg.train.shuffle
        mod = 'train'
        is_train = True
        drop_last = False
    elif cfg.mod == 'test' and not is_eval:
        batch_size = cfg.test.batch_size
        shuffle = False
        drop_last = False
        mod = 'test'
        is_train = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = False
        drop_last = False
        is_train = False
        mod = 'eval'

    dataset = make_dataset(cfg, mod)
    sampler = make_data_sampler(dataset, shuffle, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    if(is_train and cfg.sample_all == True):
        data_loader = torch.utils.data.DataLoader(dataset)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_sampler=batch_sampler,
                                                num_workers=num_workers,
                                                collate_fn=collator,
                                                worker_init_fn=worker_init_fn)
    return data_loader 
