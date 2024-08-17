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
from lib.utils.sample_utils import random_points_on_meshes_with_face_and_bary
from pytorch3d.structures import Meshes
from lib.utils.body_util import approx_gaussian_bone_volumes
from lib.utils.blend_utils import tpose_points_to_pose_points


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            if(cfg.test_novel_pose):
                test_view = [
                i for i in range(num_cams) 
                ]
            else:
                test_view = [
                    i for i in range(num_cams) if i not in cfg.train_view
                ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.train_view if split == 'train' else test_view
        i_start, i_end, i_intv = cfg.frame_range[0], cfg.frame_range[1], cfg.frame_range[2]
        
        if cfg.mod == "test":
            i_start, i_end, i_intv = cfg.test_frame_range[0], cfg.test_frame_range[1], cfg.test_frame_range[2]
            if self.human == 'CoreView_390':
                i_start = 0
        if cfg.test_novel_pose:
            i_start = cfg.test_frame_range[1]
            if(cfg.max_frame < len(annots['ims'])):
                i_end = cfg.max_frame
            else:
                i_end = len(annots['ims'])
            i_intv = cfg.test_frame_range[2]

        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i_start: i_end: i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i_start: i_end: i_intv]
        ]).ravel()
        self.num_cams = len(view)
        self.n_train_frames = len(self.ims)

        #SMPL DATA
        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy')).astype(np.int32)
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)

        
        # self.volume_weights = np.load(os.path.join(self.lbs_root, 'cano_weight_volume.npz'))['weight_volume']
        self.big_A = self.load_bigpose()
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        self.tvertices = np.load(vertices_path).astype(np.float32)
        self.faces = np.load(os.path.join(self.lbs_root, 'faces.npy')).astype(np.int64)
        if(cfg.densify_smpl):
            if not os.path.exists(os.path.join(self.data_root, 'densified_tvertices.npy')):
                meshes = Meshes(verts=torch.from_numpy(self.tvertices)[None], faces=torch.from_numpy(self.faces)[None])
                densified = random_points_on_meshes_with_face_and_bary(meshes, cfg.densify_num)
                densified_tvertices = densified[0][0].detach().cpu().numpy()
                np.save(os.path.join(self.data_root, "densified_tvertices.npy"), densified_tvertices)

        if cfg.test_novel_pose or cfg.aninerf_animation:
            training_joints_path = os.path.join(self.lbs_root, 'training_joints.npy')
            if os.path.exists(training_joints_path):
                self.training_joints = np.load(training_joints_path)
        self.nrays = cfg.N_rand


    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

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

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.jpg'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root or 'nerfcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def prepare_input(self, i):
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, cfg.params,
                    '{:06d}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        
        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(
            poses, joints, parents, return_joints=True)
        posed_joints = np.dot(canonical_joints, R.T) + Th

        # find the nearest training frame
        if (cfg.test_novel_pose or cfg.aninerf_animation) and hasattr(self, "training_joints"):
            nearest_frame_index = np.linalg.norm(self.training_joints -
                                                 posed_joints[None],
                                                 axis=2).mean(axis=1).argmin()
        else:
            nearest_frame_index = 0

        poses = poses.ravel().astype(np.float32)

        return A, Rh, Th, poses, nearest_frame_index, canonical_joints 

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # orig_msk = cv2.resize(orig_msk, (W, H),
                            #   interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        #world to cam
        R = np.array(self.cams['R'][cam_ind]).astype(np.float32)
        T = np.array(self.cams['T'][cam_ind]).astype(np.float32) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        msk = np.where(msk == 0, False, True)
        # orig_msk = cv2.resize(orig_msk, (W, H),
                            #   interpolation=cv2.INTER_NEAREST)
        if cfg.white_bg:
            img[msk == 0] = 1
        else:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif 'deepcap' in self.data_root or 'nerfcap' in self.data_root:
            i = int(os.path.basename(img_path).split('_')[-1][:-4])
            frame_index = i
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        A, Rh, Th, poses, nearest_frame_index, posed_joints = self.prepare_input(i)

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
            'rgb': img,
            'A': A,
            'mask': msk,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': self.tvertices,
            'joints': self.joints,
            'posed_joints': posed_joints,
            'faces': self.faces,
            'parents': self.parents
        }

        #world to smpl 
        Rs = cv2.Rodrigues(Rh)[0].astype(np.float32)
        S2C = getSmpl2view(R, T, Rs, Th)
        meta = {
            'S2C': S2C,
            'Rs': Rs,
            'Th': Th,
            'H': H, 
            'W': W
            }
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose or cfg.aninerf_animation:
            if 'h36m' in self.data_root:
                latent_index = 0
            else:
                latent_index = self.n_train_frames - 1
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)


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
    data_root = "data/zju_mocap/CoreView_387"
    human = "CoreView_387"
    ann_file = "data/zju_mocap/CoreView_387/annots.npy"
    from lib.utils.compare_smpl_RGB import compare_smpl_RGB
    from trimesh import Trimesh
    dataset = Dataset(data_root, human, ann_file, "train")
    for data in dataset:
        face = data['faces']
        pts = data['tvertices']
        # mesh = Trimesh(faces=face, vertices=)
        
        
        
    # data1 = dataset[10] 
    # tv = torch.from_numpy(data1['tvertices']).to('cuda:0')
    # weight_volume = CanoBlendWeightVolume(os.path.join(dataset.data_root, 'lbs', 'cano_weight_volume.npz'))
    # pbw = weight_volume.forward_weight(tv, requires_scale=True)
    # init_ppose  = tpose_points_to_pose_points(tv[None], pbw.permute(0, 2, 1), torch.from_numpy(data1['A']).to('cuda:0')[None])
    # write_ply('tmp/387.ply', init_ppose[0])
    
   