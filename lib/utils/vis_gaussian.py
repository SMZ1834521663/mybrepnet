import torch
import cv2
from lib.networks.renderer.gaussian_renderer import Renderer
def vis_gaussian(cfg, gaussians, cam, color, xyz, rotation, scaling):
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda:0")
    renderer = Renderer()
    # color = torch.ones_like(gaussians.get_xyz)
    # color = color * torch.tensor([22/255., 192/255., 166/255.], device=xyz.device)
    new_scale = scaling.clone().detach() * 1
    xyz = xyz.clone().detach() 
    ret1 = renderer.render(cfg, cam, gaussians, background, xyz.clone().detach(), override_color=color, override_rotation=rotation.clone().detach(), override_scale=new_scale.clone().detach())
    rendered_image1 = ret1["render"].permute(1, 2, 0)
    img1 =  rendered_image1[...,[2,1,0]].detach().cpu().numpy() * 255
    ret2 = renderer.render(cfg, cam, gaussians, background, gaussians.get_xyz.clone().detach(), override_color=color, override_rotation=rotation.clone().detach(), override_scale=new_scale.clone().detach())
    rendered_image2 = ret2["render"].permute(1, 2, 0)
    img2 =  rendered_image2[...,[2,1,0]].detach().cpu().numpy() * 255
    return img1, img2
    

