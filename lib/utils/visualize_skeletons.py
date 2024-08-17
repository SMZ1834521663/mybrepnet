import numpy as np
import torch
import trimesh
import cv2 as cv
import config
from lib.datasets.gaussian_dataset import Dataset

def load_ball_cylinder():
    ball = trimesh.load('ball.obj', process = False)
    cylinder = trimesh.load('cylinder.obj', process = False)
    return ball, cylinder

ball, cylinder = load_ball_cylinder()


def construct_skeletons(joints, parent_ids):
    vertices = []
    faces = []
    vertex_num = 0
    for j in range(joints.shape[0]):
        ball_v = np.array(ball.vertices).astype(np.float32)
        vertices.append(0.04 * ball_v + joints[j])
        faces.append(ball.faces + vertex_num)
        vertex_num += ball_v.shape[0]

        if parent_ids[j] >= 0:
            # add cylinder
            bone_len = np.linalg.norm(joints[j] - joints[parent_ids[j]])
            bone_d = 0.02
            cylinder_v = np.array(cylinder.vertices).astype(np.float32)
            cylinder_v[:, 1] = cylinder_v[:, 1] * bone_len / 1.0
            cylinder_v[:, [0, 2]] = cylinder_v[:, [0, 2]] * bone_d

            trans_j = np.identity(4, np.float32)
            trans_j[:3, 3] = joints[j] - np.array([0, -0.5 * bone_len, 0])
            d0 = np.array([0, 1, 0], np.float32)
            d1 = (joints[parent_ids[j]] - joints[j]) / bone_len
            cos_theta = np.dot(d0, d1)
            axis = np.cross(d0, d1)
            axis_norm = np.linalg.norm(axis)
            axis_angle = np.arccos(cos_theta) / axis_norm * axis
            rot = np.identity(4, np.float32)
            rot[:3, :3] = cv.Rodrigues(axis_angle)[0]
            rot[:3, 3] = -rot[:3, :3] @ joints[j] + joints[j]
            affine_mat = rot @ trans_j

            cylinder_v = np.einsum('ij,vj->vi', affine_mat[:3, :3], cylinder_v) + affine_mat[:3, 3]
            vertices.append(cylinder_v)
            faces.append(cylinder.faces + vertex_num)
            vertex_num += cylinder_v.shape[0]

    vertices = np.concatenate(vertices, 0)
    faces = np.concatenate(faces, 0)
    return vertices, faces


if __name__ == '__main__':
    data_root = "data/zju_mocap/CoreView_313_for_gaussian"
    human = "CoreView_313"
    ann_file = "data/zju_mocap/CoreView_313_for_gaussian/annots.npy"
    dataset = Dataset(data_root, human, ann_file, "train")
    joints = dataset.joints[:22]
    parents = dataset.parents[:22]
    parents[0] = 0
    vertices, faces = construct_skeletons(joints, parents)
    skeleton_mesh = trimesh.Trimesh(vertices, faces, process = False)
    skeleton_mesh.show()
    skeleton_mesh.export('skeleton_mesh.obj')
