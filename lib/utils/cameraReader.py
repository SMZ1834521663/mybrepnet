from typing import NamedTuple
import numpy as np
from gaussian_splatting.scene.dataset_readers import getNerfppNorm 
from gaussian_splatting.utils.graphics_utils import fov2focal, focal2fov
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from torch import nn
import torch
class Camera(nn.Module):
    def __init__(self, cam_id, R, T, FoVx, FoVy, Cx, Cy, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.cam_id = cam_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = Cx
        self.cy = Cy
        self.image_width = width
        self.image_height = height
        self.data_device = torch.device(data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans), dtype=torch.float32).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy, scale=scale, H=height, W=width).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def readCamera(cam, cfg) -> Camera:
    S2C = cam['S2C']
    R = S2C[:3, :3].T
    T = S2C[:3, 3]
    FovX = focal2fov(cam['K'][0, 0], cam['W'])
    FovY = focal2fov(cam['K'][1, 1], cam['H'])
    cx = cam['K'][0, 2] 
    cy = cam['K'][1, 2] 

    cam = Camera(cam_id = cam['cam_ind'], R = R, T = T, FoVy = FovY, FoVx = FovX, Cx = cx, Cy = cy,
                           width = cam['W'], height = cam['H'])
    return cam 
   
# def readCameraFromKRT(K, RT)->Camera:
#     S2C = RT
#     R = S2C[:3, :3].T
#     T = S2C[:3, 3]
#     FovX = focal2fov(K[:, 0, 0], 1024)
#     FovY = focal2fov(K[:, 1, 1], 1024)
#     cam = Camera(cam_id = 0, R = R, T = T, FoVy = FovY, FoVx = FovX,
#                            width = 1024, height = 1024)
#     return cam
 

def getNerfppNorm(cams):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cams:
        W2C = getWorld2View2(cam[:3, :3], cam[:3 ,3])
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def getAllcamNorm(path):
    cams = np.load(path, allow_pickle=True).item()['cams']
    R = np.array(cams['R'])
    T = np.array(cams['T']) / 1000.
    RT = np.concatenate([R,T], axis = -1)
    return getNerfppNorm(RT)

