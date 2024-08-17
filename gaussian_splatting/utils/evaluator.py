import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
from termcolor import colored
import torch
from lib.losses.lpips.lpips import LPIPS


class Evaluator:
    def __init__(self):
        self.mse = []
        self.lpips = []
        self.psnr = []
        self.ssim = []
        self.lpips_net = LPIPS(net='vgg').cuda()

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def scale_for_lpips(self, image_tensor):
        return image_tensor * 2. - 1.

    def get_loss(self, rgb, target):
        rgb = rgb.cuda()
        target = target.cuda()
        lpips_loss = self.lpips_net(self.scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                        self.scale_for_lpips(target.permute(0, 3, 1, 2)))
        return torch.mean(lpips_loss).cpu().detach().numpy()

    def ssim_metric(self, rgb_pred, rgb_gt, batch, mask = None):
               # convert the pixels into an image
        img_pred = rgb_pred
        img_gt = rgb_gt
        # if(mask is not None):
        #     mask_at_box = mask.detach().cpu().numpy()
        #     H, W = rgb_gt.shape
        #     mask_at_box = mask_at_box.reshape(H, W)
        #     img_pred = np.zeros((H, W, 3))
        #     img_pred[mask_at_box] = rgb_pred
        #     img_gt = np.zeros((H, W, 3))
        #     img_gt[mask_at_box] = rgb_gt
        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index']
        view_index = batch['cam']['cam_ind']
        # msk = batch['msk'][0].detach().cpu().numpy()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))
        ssim = compare_ssim(img_pred, img_gt, channel_axis=2, data_range=1)
        return ssim

    def evaluate(self, rgb_pred, rgb_gt, batch, mask = None):
        rgb_pred = rgb_pred.detach().cpu().numpy()
        rgb_gt = rgb_gt.detach().cpu().numpy()

        if rgb_gt.sum() == 0:
            return

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch, mask)
        self.ssim.append(ssim)
        lpips_loss = self.get_loss(rgb=torch.from_numpy(rgb_pred).float().unsqueeze(0), target=torch.from_numpy(rgb_gt).float().unsqueeze(0))
        self.lpips.append(lpips_loss)

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        np.save(result_path, metrics)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpips:{}'.format(np.mean(self.lpips)))
        if cfg.test_novel_pose:
            log = f"novel pose: {cfg.result_dir} mse: {np.mean(self.mse)} \n psnr: {np.mean(self.psnr)} \n ssim: {np.mean(self.ssim)}\n lpips:{(np.mean(self.lpips))}\n"
        else:
            log = f"{cfg.result_dir} mse: {np.mean(self.mse)} \n psnr: {np.mean(self.psnr)} \n ssim: {np.mean(self.ssim)}\n lpips:{(np.mean(self.lpips))}\n"
        with open(f"log/test_log/{cfg.exp_name}.txt", 'a') as f:
            f.write(log)
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips= []
        result = {"mse" : self.mse, "psnr" : self.psnr, "ssim" : self.ssim, "lpips": self.lpips}
      
        return result

