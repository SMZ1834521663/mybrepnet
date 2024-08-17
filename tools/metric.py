import numpy as np
import os

exp_name = 'delete_neuralbody'
metric_pth = os.path.join('data/result/deform/{}/metrics.npy'.format(exp_name))
m = np.load(metric_pth, allow_pickle=True).item()
print(np.mean(m['psnr']))
print(np.mean(m['ssim']))
print(np.mean(m['mse']))


       