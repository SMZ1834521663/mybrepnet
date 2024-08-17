import cv2 
import torch
from plyfile import PlyData,PlyElement
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# from lib.utils.if_nerf 
import cv2
from lib.utils import base_utils
import matplotlib.pyplot as plt

from lib.utils.plotting import cv_colors
Kintree = {
   'kintree':[
    [1, 0], [2, 0], [3, 0], [4, 1], [5, 2], [6, 3], [7, 4], [8,5],
    [9, 6], [10, 7], [11, 8], [12, 9], [13, 9], [14, 9], [15, 12], [16, 13],
    [17, 14], [18, 16], [19, 17], [20, 18], [21, 19], [22, 20],[23, 21]
    ],
    'color':[
        'k', 'r', 'r', 'r', 'b', 'b', 'b', 'k', 'r', 'r', 'r', 'b', 'b', 'b',
        'y', 'y', 'y', 'y', 'b', 'b', 'b', 'r', 'r'
    ]
}

def write_ply(save_path, points, text = True):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text = text).write(save_path)

def vis_rays_3d(v, ray_o, ray_d, size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(ray_o.shape[1]):
        x = v[0, i, :, 0]
        y = v[0, i, :, 1]
        z = v[0, i, :, 2]
        rx = ray_o[0, i, 0]
        ry = ray_o[0, i, 1]
        rz = ray_o[0, i, 2]
        ax.scatter(x,y,z, s=size)
        ax.scatter(rx, ry, rz, s = 20, c = 'Green')
    plt.show()

def vis_3d(v):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]
    ax.scatter(x,y,z)
    plt.show()

def plotSkel3D(pts,
               config = Kintree,
               ax=None,
               phi=0,
               theta=0,
               max_range=1,
               linewidth=4,
               color=None):
    multi = False
    if torch.is_tensor(pts):
        if len(pts.shape) == 3:
            print(">>> Visualize multiperson ...")
            multi = True
            if pts.shape[1] != 3:
                pts = pts.transpose(1, 2)
        elif len(pts.shape) == 2:
            if pts.shape[0] != 3:
                pts = pts.transpose(0, 1)
        else:
            raise RuntimeError('The dimension of the points is wrong!')
        pts = pts.detach().cpu().numpy()
    else:
        print(pts.shape)
        if pts.shape[0] != 3:
            pts = pts.T
    # pts : bn, 3, NumOfPoints or (3, N)
    if ax is None:
        print('>>> create figure ...')
        fig = plt.figure(figsize=[5,5])
        ax = fig.add_subplot(111, projection='3d')
    for idx, (i, j) in enumerate(config['kintree']):
        if multi:
            for b in range(pts.shape[0]):
                ax.plot([pts[b][0][i], pts[b][0][j]],
                        [pts[b][1][i], pts[b][1][j]],
                        [pts[b][2][i], pts[b][2][j]],
                        lw=linewidth,
                        color=config['color'][idx] if color is None else color,
                        alpha=1)
        else:
            ax.plot([pts[0][i], pts[0][j]], [pts[1][i], pts[1][j]],
                    [pts[2][i], pts[2][j]],
                    lw=linewidth,
                    color=config['color'][idx],
                    alpha=1)
    if multi:
        for b in range(pts.shape[0]):
            ax.scatter(pts[b][0], pts[b][1], pts[b][2], color='r', alpha=1)
    else:
        ax.scatter(pts[0], pts[1], pts[2], color='r', alpha=1, s=0.5)
    ax.view_init(phi, theta)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return ax

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def plot_bounding_box(bound, K, pose):
    corners_3d = get_bound_corners(bound)
    box_lines = np.zeros([12, 100, 3])
    box_lines = np.stack(
        [np.linspace(corners_3d[0], corners_3d[1], 100),
        np.linspace(corners_3d[0], corners_3d[2], 100),
        np.linspace(corners_3d[0], corners_3d[4], 100),
        np.linspace(corners_3d[1], corners_3d[3], 100),
        np.linspace(corners_3d[1], corners_3d[5], 100),
        np.linspace(corners_3d[2], corners_3d[3], 100),
        np.linspace(corners_3d[2], corners_3d[6], 100),
        np.linspace(corners_3d[3], corners_3d[7], 100),
        np.linspace(corners_3d[4], corners_3d[5], 100),
        np.linspace(corners_3d[4], corners_3d[6], 100),
        np.linspace(corners_3d[5], corners_3d[7], 100),
        np.linspace(corners_3d[6], corners_3d[7], 100)]
    )
    box_lines = box_lines.reshape(-1, 3)
    box_points_2d = base_utils.project(box_lines, K, pose)
    return box_points_2d, box_lines
    


    
    

