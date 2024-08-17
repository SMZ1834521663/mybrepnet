import sys
sys.path.append('/home/wzx2021/gaussian-human/')
import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.vis_3d import *
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getSmpl2view
import pickle

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def get_camera(camera_path):
    camera = read_pickle(camera_path)
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    camera = {'K': K, 'R': R, 'T': T, 'D': D}
    return camera

def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        # smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["body_pose"] = smpl_params["thetas"]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

class Dataset(data.Dataset):
    def __init__(self, num_frames, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        self.num_frames = num_frames

        annots = np.load(ann_file, allow_pickle=True).item()
        self.annots = annots
        self.params = np.load(os.path.join(data_root, 'params.npy'), allow_pickle=True).item()
        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = get_camera(camera_path)
        self.i_start, self.i_end, self.i_intv = cfg.frame_range[0], cfg.frame_range[1], cfg.frame_range[2]
        self.frame_index = 0
        self.num_cams = 1
        cached_path = os.path.join(self.data_root, "poses/anim_nerf_train.npz")
        if cached_path and os.path.exists(cached_path):
            print(f"val Loading from", cached_path)
            self.smpl_params = load_smpl_param(cached_path)


        #SMPL DATA
        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy')).astype(np.int32)
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        self.tvertices = np.load(vertices_path).astype(np.float32)
        self.faces = np.load(os.path.join(self.lbs_root, 'faces.npy')).astype(np.int64)
        self.nrays = cfg.N_rand

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def prepare_input(self, i):
        index = self.frame_index
        Rh = self.smpl_params['global_orient'][index]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = self.smpl_params['transl'][index].astype(np.float32)
        poses = np.concatenate([Rh.reshape(1, 3), self.smpl_params['body_pose'][index].reshape(-1, 3)], axis = 0)
        joints = self.joints
        parents = self.parents
        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
            poses, joints, parents, return_joints=True)
        posed_joints = np.dot(canonical_joints, R.T) + Th
        poses = poses.ravel().astype(np.float32)
        return A, Rh, Th, poses, canonical_joints

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'image',
                                '{}.jpg'.format(self.frame_index))
        img = imageio.imread(img_path).astype(np.float32) / 255.
        H, W = img.shape[:2]
        K = self.cam['K']
        D = self.cam['D']
        img = cv2.undistort(img, K, D)
        R = self.cam['R'].astype(np.float32)
        T = self.cam['T'][:, None].astype(np.float32)
        RT = np.concatenate([R, T], axis=1).astype(np.float32)
        A, Rh, Th, poses, posed_joints = self.prepare_input(self.frame_index)
        Rs = cv2.Rodrigues(Rh)[0].astype(np.float32)
        angle = 2 * np.pi * index / self.num_frames
        R_rot = cv2.Rodrigues(np.array([0, angle, 0]))[0].astype(np.float32)
        R_gt = R_rot @ Rs

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        K[:2] = K[:2] * cfg.ratio

        camera = {
            'K': K,
            'R': R, #world2cam
            'D': D,
            'T': T, #world2cam 'cam_ind': cam_ind,
            'W': W,
            'H': H
        }
        ret = {
            'cam': camera,
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': self.tvertices,
            'joints': self.joints,
            'posed_joints': posed_joints,
            'faces': self.faces,
            'parents': self.parents
        }

        S2C = getSmpl2view(R, T, R_gt, Th)
        meta = {
            'S2C': S2C,
            'Rs': Rs,
            'R_gt': R_gt,
            'Th': Th,
            'H': H, 
            'W': W
            }
        ret.update(meta)

        latent_index = 0
        bw_latent_index = 0
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': self.frame_index,
            'cam_ind': index
        }
        ret.update(meta)
        return ret

    def __len__(self):
        return self.num_frames


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts

#unit test
if __name__ == "__main__":
    import torch.nn.functional as F
    from lib.utils.blend_utils import tpose_points_to_pose_points
    from lib.networks.volume import CanoBlendWeightVolume
    data_root = "data/people_snapshot_public/female-3-casual"
    human = "female-3-casual"
    ann_file = "data/people_snapshot_public/female-3-casual/annots.npy"
    dataset = Dataset(data_root, human, ann_file, "test")
    # for data in dataset:
        
     