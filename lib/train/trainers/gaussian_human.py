import torch.nn as nn
import datetime
import torch
from gaussian_splatting.utils.general_utils import build_rotation
from lib.networks.renderer.gaussian_renderer import Renderer
from gaussian_splatting.scene import GaussianModel
from lib.train.recorder import Recorder
from lib.utils.cameraReader import readCamera, getAllcamNorm
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from lib.evaluators import Evaluator 
from lib.utils.vis_3d import *
import time
import os
import cv2
import numpy as np
from lib.utils.gaussian_utils import *
from lib.utils.blend_utils import pose_dir_to_can_dir
from lib.utils.sample_utils import random_sample
from gaussian_splatting.utils.image_utils import *
from line_profiler import LineProfiler
import tqdm
from lib.losses.lpips.lpips import LPIPS
from lib.utils.vis_gaussian import vis_gaussian
from lib.utils.if_nerf import if_nerf_data_utils
from PIL import Image
from random import randint

def to_cuda(batch, device):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b) for b in batch]
        return batch

    for k in batch:
        if k == 'meta' or k == 'cam':
            continue
        if isinstance(batch[k], tuple) or isinstance(batch[k], list):
            batch[k] = [torch.from_numpy(b).to(device) for b in batch[k]]
        elif isinstance(batch[k], dict):
            for b in batch[k]:
                if(isinstance(batch[k][b], np.ndarray)):
                    batch[k][b] = torch.from_numpy(batch[k][b]).to(device)
        else:
            if(isinstance(batch[k], np.ndarray)):
                batch[k] = torch.from_numpy(batch[k]).to(device)
    return batch

def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class GaussianHuman:
    def __init__(self, cfg, scene_data):
        self.device = torch.device('cuda:0')
        self.renderer = Renderer()
        self.cfg = cfg
        if(self.cfg.densify_smpl):
            vertices_path = os.path.join(self.cfg.data_root, 'densified_tvertices.npy')
        else:
            vertices_path = os.path.join(self.cfg.data_root, 'lbs', 'tvertices.npy')
        self.tvertices = torch.tensor(scene_data['tvertices'], device=self.device)
        self.weights = torch.tensor(scene_data['weights'], device=self.device)
        self.joints = torch.tensor(scene_data['joints'], device=self.device)
        self.faces = torch.tensor(scene_data['faces'], device=self.device)
        self.parents = torch.tensor(scene_data['parents'], device=self.device)
        self.big_A = torch.tensor(scene_data['big_A'], device=self.device)
        self.tbounds = torch.tensor(scene_data['tbounds'], device=self.device)
        self.img2l1 = lambda x, y: torch.abs(x-y).mean()
        cam_path = os.path.join(cfg.data_root, "annots.npy")
        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.bg_color = [1, 1, 1] if self.cfg.white_bg else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device=self.device)
        self.nerf_normalization = getAllcamNorm(cam_path)
        if cfg.mod == 'train':
            self.nerf_normalization['radius'] = 1
        self.smpl_utils = {
            'weights': self.weights,
            'tvertices': self.tvertices,
            'faces': self.faces,
            'joints': self.joints,
            'parents': self.parents,
            'big_A': self.big_A,
            'tbounds': self.tbounds
        }
        if self.cfg.snapshot:
            self.nerf_normalization['radius'] = 1.45
        self.gaussians = GaussianModel(self.cfg, self.cfg.sh_degree, self.device)
        if(self.cfg.lpips_loss):
            self.lpips = LPIPS(net='vgg').cuda()
            set_requires_grad(self.lpips, requires_grad=False)

    def reset_knn(self):
        self.neighbor_sq_dist, self.neighbor_indices = o3d_knn(self.gaussians.get_xyz.detach().cpu().numpy(), self.cfg.num_knn)
        self.neighbor_indices = torch.tensor(self.neighbor_indices).to(self.device).long().contiguous()
        self.neighbor_weight = np.exp(-2000 * self.neighbor_sq_dist)
        self.neighbor_weight = torch.tensor(self.neighbor_weight).to(self.device).float().contiguous()
        self.neighbor_dist = np.sqrt(self.neighbor_sq_dist)
        self.neighbor_dist = torch.tensor(self.neighbor_dist).to(self.device).float().contiguous()
    
    def load(self, model_args, training_args):
        self.gaussians.restore(model_args, training_args)
        self.reset_knn()

    def forward_once(self, view, iter):
        cam = readCamera(view['cam'], self.cfg)
        cam_center = cam.camera_center
        pxyz, resd, prot, res_rot, new_scale, res_scale, pose_latent, R = self.gaussians.deform(view, self.smpl_utils, iter)
        view_dir = pxyz - cam_center
        pt_wise_view = view_dir / torch.norm(view_dir, dim = -1, keepdim = True)
        pt_wise_view = pt_wise_view.reshape(-1, 3)
      
        if self.cfg.debug:
            write_ply(f'tmp/posed_{iter}.ply', pxyz[0])
            write_ply(f'tmp/cam_center_{iter}.ply', cam_center[None])
        if self.cfg.can_view:
            R_sc = torch.from_numpy(view['cam']['S2C'][:3, :3]).to(pt_wise_view)
            pt_wise_view = pt_wise_view @ R_sc.T
            pt_wise_view = torch.sum(R[0].transpose(1,2) * pt_wise_view[..., None], dim=-1)
        if self.cfg.rgb_feature or self.cfg.rgb_only:
            if(self.cfg.pose_color):
                color = self.gaussians.compute_color(view, pt_wise_view, self.gaussians.get_xyz, self.smpl_utils, pose_latent)
            else:
                color = self.gaussians.compute_color(view, pt_wise_view, self.gaussians.get_xyz, self.smpl_utils)
            ret = self.renderer.render(self.cfg, cam, self.gaussians, self.background, pxyz[0], override_color=color, override_rotation=prot[0], override_scale=new_scale[0])
        else:
            ret = self.renderer.render(self.cfg, cam, self.gaussians, self.background, pxyz[0], override_rotation = prot[0], override_scale = new_scale[0])

        if(self.cfg.debug_gaussian):
            gaussianp, gaussiant = vis_gaussian(self.cfg, self.gaussians, cam, color, pxyz[0], prot[0], new_scale[0]) 
        #     return gaussianp, gaussiant
        rendered_image = ret["render"].permute(1, 2, 0)
        screen_points = ret["wiewspace_points"]
        vis = ret["visibility_filter"]
        radii = ret["radii"]
        scalar_stats = {}
        loss = 0
        if(self.cfg.mod == 'train'):
            gt_image = view['rgb']
            mask = view['mask'][..., None].float()
            rgb_loss = self.img2l1(rendered_image, gt_image) 
            loss += self.cfg.lambda_rgb * rgb_loss
            if self.cfg.ssim_loss:
                loss += self.cfg.lambda_dssim * (1 - ssim(rendered_image, gt_image))

            if self.cfg.bound_mask:
                bound_mask = view['bound_mask']
                x_min, y_min, x_max, y_max = bound_mask
                gt_image = gt_image[x_min:x_max, y_min:y_max]
                rendered_image = rendered_image[x_min:x_max, y_min:y_max] 

            if self.cfg.lpips_loss:
                lpips_loss = self.lpips(scale_for_lpips(rendered_image[None].permute(0, 3, 1, 2)), 
                                    scale_for_lpips(gt_image[None].permute(0, 3, 1, 2)))
                lpips_loss = torch.mean(lpips_loss)  
                scalar_stats.update({'lpips_loss': lpips_loss})
                loss += lpips_loss * self.cfg.lambda_lpips


            if self.cfg.mask_loss:
                rendered_mask = ret['rendered_mask'].permute(1, 2, 0)
                mask_loss = self.img2l1(rendered_mask, mask)
                # mask_image = rendered_mask[...,[2,1,0]].detach().cpu().numpy() * 255
                # cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_mask.jpg', mask_image)
                # cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_mask_binary.jpg', mask.detach().cpu().numpy() * 255)
                loss += self.cfg.lambda_mask_loss * mask_loss 
                scalar_stats.update({'mask_loss': mask_loss})
     
           
            if self.cfg.local_rigid:
                neighbor_pts = pxyz[0][self.neighbor_indices]
                curr_offset = neighbor_pts - pxyz[0][:, None]
                curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
                if self.cfg.rigid_loss:
                    can_rot = self.gaussians.get_rotation
                    can_xyz = self.gaussians.get_xyz
                    prot_inv = prot[0].clone()
                    can_offset = can_xyz[self.neighbor_indices] - can_xyz[:,None]
                    prot_inv[:, 1:] = -1 * prot_inv[:, 1:]
                    rel_rot = quat_mult(can_rot, prot_inv)
                    rot = build_rotation(rel_rot)
                    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
                    rigid_loss = weighted_l2_loss_v2(curr_offset_in_prev_coord, can_offset,
                                                self.neighbor_weight)
                    loss += self.cfg.lambda_rigid_loss * rigid_loss
                    scalar_stats.update({'rigid_loss': rigid_loss})

                if self.cfg.rot_loss:
                    neighbor_rot = prot[0][self.neighbor_indices]
                    neighbor_rot_dis = weighted_rot_dis(neighbor_rot, prot[0], self.neighbor_weight)
                    loss += self.cfg.lambda_rot_loss * neighbor_rot_dis
                    scalar_stats.update({'neighbor_rot_loss': neighbor_rot_dis})

                    
                iso_loss = weighted_l2_loss_v1(curr_offset_mag, self.neighbor_dist, self.neighbor_weight)
                loss += self.cfg.lambda_iso * iso_loss
                scalar_stats.update({'iso_loss': iso_loss})

            if self.cfg.reg_resd:
                reg_resd = torch.norm(resd, dim=-1).mean()
                loss += self.cfg.lambda_reg_resd * reg_resd
                scalar_stats.update({'offset_loss': reg_resd})
        
            if self.cfg.reg_res_rot:
                reg_res_rot = torch.norm(res_rot).mean()
                loss += reg_res_rot * self.cfg.lambda_reg_res_rot
                scalar_stats.update({'reg_res_rot': reg_res_rot})

            if self.cfg.reg_res_scale:
                reg_res_scale = torch.norm(res_scale).mean()
                loss += reg_res_scale * self.cfg.lambda_reg_res_scale
                scalar_stats.update({'reg_res_scale': reg_res_scale})
            
            scalar_stats.update({'rgb_loss': rgb_loss})
            scalar_stats.update({'loss': loss})

        if(self.cfg.debug_gaussian):
            return loss, scalar_stats, rendered_image, screen_points, vis, radii, gaussianp, gaussiant
        return loss, scalar_stats, rendered_image, screen_points, vis, radii

    def train(self, first_iter, data, recorder: Recorder):
        max_iter = self.cfg.iterations
        end = time.time()
        for iter in tqdm.tqdm(range(first_iter, max_iter + 1)):
            view_ = data[randint(0, len(data) - 1)].copy()
            view_ = to_cuda(view_, self.device)
            if(iter == 1):
                if(not self.cfg.resume):
                    if(self.cfg.random_init):
                        bound_min, _ = torch.min(self.tvertices, dim=0)
                        bound_max, _ = torch.max(self.tvertices, dim=0)
                        bounds = torch.stack([bound_min, bound_max], dim=0)
                        init_vertex = random_sample(bounds, self.cfg.points_num)
                    else:
                        init_vertex = self.tvertices
                
                self.gaussians.initialize_from_human(init_vertex, self.nerf_normalization["radius"])
                self.gaussians.training_setup(self.cfg)
                if(self.cfg.local_rigid):
                    self.reset_knn()
            # if iter == self.cfg.optimize_gaussian_until:
                # self.gaussians.frozen()
            data_time = time.time() - end
            if iter % 5000 == 0:
                self.gaussians.oneupSHdegree()
            if self.cfg.debug_gaussian:
                loss, scalar_stats, rendered_image, screen_points, vis, radii, gaussianp, gaussiant = self.forward_once(view_, iter)
            else:
                loss, scalar_stats, rendered_image, screen_points, vis, radii = self.forward_once(view_, iter)
            loss.backward()
            self.gaussians.update_learning_rate(iter) 
            if iter % 50 == 0 and self.cfg.vis_log:
                frame_index = view_['frame_index']
                if self.cfg.debug_gaussian:
                    cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_{frame_index}_gaussianp.jpg',gaussianp)
                    cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_{frame_index}_gaussiant.jpg',gaussiant)
                img = rendered_image[...,[2,1,0]].detach().cpu().numpy() * 255
                cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_{frame_index}_rendered.jpg',img)
                cv2.imwrite(f'log/{self.cfg.exp_name}/{iter}_{frame_index}_gt.jpg', view_['rgb'][...,[2,1,0]].detach().cpu().numpy() * 255)
                
            with torch.no_grad():
                if iter < self.cfg.densify_until_iter:
                    self.gaussians.max_radii2D[vis] = torch.max(self.gaussians.max_radii2D[vis], radii[vis])
                    self.gaussians.add_densification_stats(screen_points, vis)

                    if iter > self.cfg.densify_from_iter and iter % self.cfg.densification_interval == 0:
                        size_threshold = 20 if iter > self.cfg.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.cfg.densify_grad_threshold, 0.005, self.nerf_normalization["radius"],  size_threshold)
                        print(f"densify gaussians to {self.gaussians.get_xyz.shape[0]}")
                        if self.cfg.local_rigid:
                            self.reset_knn()
                    if iter % self.cfg.opacity_reset_interval == 0 or (self.cfg.white_bg and iter == self.cfg.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iter < max_iter:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

            batch_time = time.time() - end
            end = time.time()
            recorder.step += 1
            recorder.update_loss_stats(scalar_stats)
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            if iter % self.cfg.log_interval == 0:
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                num_gaussians = str(self.gaussians.get_xyz.shape[0])
                training_state = f"exp_name: {self.cfg.exp_name}, remaining training time:{eta_string}, {str(recorder)}, num_gaussians:{num_gaussians}"
                print(training_state)

            if iter % self.cfg.record_interval == 0:
                image_stats = {"img": rendered_image.permute(2, 0, 1)}
                recorder.update_image_stats(image_stats)
                recorder.record('train')

            if iter % self.cfg.save_ckpt_interval == 0:
                print(f"{iter} iterations saving checkpoint")
                if(not os.path.exists(self.cfg.trained_model_dir)):
                    os.makedirs(self.cfg.trained_model_dir)
                torch.save((self.gaussians.capture(), iter), self.cfg.trained_model_dir + "/chkpnt" + str(iter) + ".pth")
                torch.save((self.gaussians.capture(), iter), self.cfg.trained_model_dir + "/chkpnt" + "latest.pth")


    def val(self, test_data, recorder: Recorder):
        from gaussian_splatting.utils.evaluator import Evaluator
        end = time.time()
        self.gaussians.eval(self.device) 
        evaluator = Evaluator()
        i = 0
        for view in tqdm.tqdm(test_data):
            view = to_cuda(view, self.device) 
            data_time = time.time() - end
            with torch.no_grad():
                # lp = LineProfiler()
                if self.cfg.debug_gaussian:
                    gast, gasp = self.forward_once(view, self.cfg.optimize_non_rigid_from + 1) 
                    frame_index = ['frame_index']
                    view_index = view['cam_ind']
                    cv2.imwrite(
                            'gaussians/frame{:04d}_view{:04d}.png'.format(frame_index,
                                                    view_index), gast)
                    cv2.imwrite(
                            'gaussians/frame{:04d}_view{:04d}p.png'.format(frame_index,
                                                    view_index), gasp)
                    i = i + 1
                else:
                    loss, scalar_stats, rendered_image, screen_points, vis, radii= self.forward_once(view, self.cfg.optimize_non_rigid_from + 1)
                    evaluator.evaluate(rendered_image, view['rgb'], view)
        evaluator.summarize()

#test_novel_pose
    def animate(self, test_data, poses, recorder: Recorder):
        self.gaussians.eval(self.device) 

        os.mkdir(f'animate/{self.cfg.exp_name}')
        with torch.no_grad():
            for view in test_data:
                i = 0
                for pose in poses:
                    view['poses'] = pose.reshape(-1)
                    view = to_cuda(view, self.device) 
                    cam = readCamera(view['cam'], self.cfg)
                    cam_center = cam.camera_center
                    pxyz, resd, prot, res_rot, new_scale, res_scale, pose_latent, R = self.gaussians.deform(view, self.smpl_utils, iter)
                    view_dir = pxyz - cam_center
                    pt_wise_view = view_dir / torch.norm(view_dir, dim = -1, keepdim = True)
                    pt_wise_view = pt_wise_view.reshape(-1, 3)
                    if self.cfg.can_view:
                        R_sc = torch.from_numpy(view['cam']['S2C'][:3, :3]).to(pt_wise_view)
                        pt_wise_view = pt_wise_view @ R_sc.T
                        pt_wise_view = torch.sum(R[0].transpose(1,2) * pt_wise_view[..., None], dim=-1)
                    if self.cfg.rgb_feature or self.cfg.rgb_only:
                        if(self.cfg.pose_color):
                            color = self.gaussians.compute_color(view, pt_wise_view, self.gaussians.get_xyz, self.smpl_utils, pose_latent)
                        else:
                            color = self.gaussians.compute_color(view, pt_wise_view, self.gaussians.get_xyz, self.smpl_utils)
                        ret = self.renderer.render(self.cfg, cam, self.gaussians, self.background, pxyz[0], override_color=color, override_rotation=prot[0], override_scale=new_scale[0])
                    else:
                        ret = self.renderer.render(self.cfg, cam, self.gaussians, self.background, pxyz[0], override_rotation = prot[0], override_scale = new_scale[0])
                    rendered_image = ret["render"].permute(1, 2, 0)
                    rendered_image = rendered_image[...,[2,1,0]].detach().cpu().numpy() * 255
                    # frame_index = view['frame_index']
                    cam_ind = view['cam']['cam_ind']
                    cv2.imwrite(f'animate/{self.cfg.exp_name}/view_{cam_ind}_{i}.jpg', rendered_image)
                    i+=1
        
    

    def free_view(self, data_loader):
        end = time.time()
        self.gaussians.eval(self.device) 
        i = 0
        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch, self.device) 
            data_time = time.time() - end
            with torch.no_grad():
                if self.cfg.debug_gaussian:
                    gast, gasp = self.forward_once(batch, self.cfg.optimize_non_rigid_from + 1) 
                    frame_index = batch['frame_index'].item()
                    view_index = batch['cam_ind'].item()
                    cv2.imwrite(
                            'gaussians/frame{:04d}_view{:04d}.png'.format(frame_index,
                                                    view_index), gast)
                    cv2.imwrite(
                            'gaussians/frame{:04d}_view{:04d}p.png'.format(frame_index,
                                                    view_index), gasp)
                    i = i + 1
                else:
                    loss, scalar_stats, rendered_image, screen_points, vis, radii= self.forward_once(batch, self.cfg.optimize_non_rigid_from + 1)
                    frame_index = batch['frame_index'].item()
                    view_index = batch['cam_ind'].item()
                    img = rendered_image[...,[2,1,0]].detach().cpu().numpy() * 255
                    cv2.imwrite(
                            'freeview/frame{:04d}_view{:04d}.png'.format(frame_index,
                                                    view_index), img)
                  

  
            


  