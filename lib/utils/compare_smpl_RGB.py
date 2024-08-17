# import numpy as np
# import torch
from lib.utils.render_mesh import Renderer
# import trimesh
import cv2

def compare_smpl_RGB(smpl_mesh, img, K, R, T,  frame_index, cam_index):
    H,W = img.shape[:2]
    img = img[...,[2,1,0]]
    renderer = Renderer(height = H, width = W)
    cv2.imwrite(f'mesh_rendered/{frame_index}_{cam_index}.jpg', img * 255)
    render_mesh_img = renderer.render(smpl_mesh, K, R, T)[..., [2, 1, 0]]
    # mask = render_mesh_img[render_mesh_img != 0]
    # img[mask] = 0
    # cv2.imwrite('img.jpg', img * 255)
    cv2.imwrite(f'mesh_rendered/{frame_index}__{cam_index}.png', render_mesh_img)
