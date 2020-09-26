"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import torch.utils.data as data
import torch
import numpy as np
import os
import pickle
import random
from scipy.signal import savgol_filter
from util.icp import icp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from scipy.linalg import logm

STD_FACE_LANDMARK_FILE_DIR = 'dataset/utils/STD_FACE_LANDMARKS.txt'


class Audio2landmark_Dataset(data.Dataset):

    def __init__(self, dump_dir, dump_name, num_window_frames, num_window_step, status, noautovc=''):
        self.dump_dir = dump_dir
        self.num_window_frames = num_window_frames
        self.num_window_step = num_window_step

        # Step 1 : load A / V data from dump files
        print('Loading Data {}_{}'.format(dump_name, status))

        with open(os.path.join(self.dump_dir, '{}_{}_{}au.pickle'.format(dump_name, status, noautovc)), 'rb') as fp:
            self.au_data = pickle.load(fp)
        with open(os.path.join(self.dump_dir, '{}_{}_{}fl.pickle'.format(dump_name, status, noautovc)), 'rb') as fp:
            self.fl_data = pickle.load(fp)

        valid_idx = list(range(len(self.au_data)))

        random.seed(0)
        random.shuffle(valid_idx)
        self.fl_data = [self.fl_data[i] for i in valid_idx]
        self.au_data = [self.au_data[i] for i in valid_idx]

        # # normalize fls
        # for i in range(len(self.fl_data)):
        #     shape_3d = self.fl_data[i][0].reshape((-1, 68, 3))
        #     scale = np.abs(1.0 / (shape_3d[:, 36:37, 0:1] - shape_3d[:, 45:46, 0:1]))
        #     shift = - 0.5 * (shape_3d[:, 36:37] + shape_3d[:, 45:46])
        #     shape_3d = (shape_3d + shift) * scale
        #     self.fl_data[i] = (shape_3d.reshape(-1, 204), self.fl_data[i][1])

        # tmp = [au for au, info in self.au_data]
        # tmp = np.concatenate(tmp, axis=0)
        # au_mean, au_std = np.mean(tmp, axis=0), np.std(tmp, axis=0)
        # np.savetxt('dataset/utils/MEAN_STD_NOAUTOVC_AU.txt', np.concatenate([au_mean, au_std], axis=0).reshape(-1))
        # print(tmp.shape)
        # exit(0)


        au_mean_std = np.loadtxt('dataset/utils/MEAN_STD_NOAUTOVC_AU.txt') # np.mean(self.au_data[0][0]), np.std(self.au_data[0][0])
        au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

        self.au_data = [((au - au_mean) / au_std, info) for au, info in self.au_data]


    def __len__(self):
        return  len(self.fl_data)

    def __getitem__(self, item):
        # print('-> get item {}: {} {}'.format(item, self.fl_data[item][1][0], self.fl_data[item][1][1]))
        return self.fl_data[item], self.au_data[item]

    def my_collate_in_segments(self, batch):
        fls, aus, embs = [], [], []
        for fl, au in batch:
            fl_data, au_data, emb_data = fl[0], au[0], au[1][2]
            assert (fl_data.shape[0] == au_data.shape[0])

            fl_data = torch.tensor(fl_data, dtype=torch.float, requires_grad=False)
            au_data = torch.tensor(au_data, dtype=torch.float, requires_grad=False)
            emb_data = torch.tensor(emb_data, dtype=torch.float, requires_grad=False)

            # window shift data
            fls += [fl_data[i:i + self.num_window_frames] #- fl_data[i]
                    for i in range(0, fl_data.shape[0] - self.num_window_frames, self.num_window_step)]
            aus += [au_data[i:i + self.num_window_frames]
                    for i in range(0, au_data.shape[0] - self.num_window_frames, self.num_window_step)]
            embs += [emb_data] * ((au_data.shape[0] - self.num_window_frames) // self.num_window_step)

        # fls = torch.tensor(fls, dtype=torch.float, requires_grad=False)
        # aus = torch.tensor(aus, dtype=torch.float, requires_grad=False)
        # embs = torch.tensor(embs, dtype=torch.float, requires_grad=False)

        fls = torch.stack(fls, dim=0)
        aus = torch.stack(aus, dim=0)
        embs = torch.stack(embs, dim=0)

        return fls, aus, embs

    def my_collate_in_segments_noemb(self, batch):
        fls, aus, embs = [], [], []
        for fl, au in batch:
            fl_data, au_data = fl[0], au[0]
            assert (fl_data.shape[0] == au_data.shape[0])

            fl_data = torch.tensor(fl_data, dtype=torch.float, requires_grad=False)
            au_data = torch.tensor(au_data, dtype=torch.float, requires_grad=False)

            # window shift data
            fls += [fl_data[i:i + self.num_window_frames]  # - fl_data[i]
                    for i in range(0, fl_data.shape[0] - self.num_window_frames, self.num_window_step)]
            aus += [au_data[i:i + self.num_window_frames]
                    for i in range(0, au_data.shape[0] - self.num_window_frames, self.num_window_step)]

        fls = torch.stack(fls, dim=0)
        aus = torch.stack(aus, dim=0)

        return fls, aus


def estimate_neck(fl):
    mid_ch = (fl[2, :] + fl[14, :]) * 0.5
    return (mid_ch * 2 - fl[33, :]).reshape(1, 3)

def norm_output_fls_rot(fl_data_i, anchor_t_shape=None):

    # fl_data_i = savgol_filter(fl_data_i, 21, 3, axis=0)

    t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
    if(anchor_t_shape is None):
        anchor_t_shape = np.loadtxt(
            r'dataset/utils/ANCHOR_T_SHAPE_{}.txt'.format(len(t_shape_idx)))
        s = np.abs(anchor_t_shape[5, 0] - anchor_t_shape[8, 0])
        anchor_t_shape = anchor_t_shape / s * 1.0
        c2 = np.mean(anchor_t_shape[[4,5,8], :], axis=0)
        anchor_t_shape -= c2

    else:
        anchor_t_shape = anchor_t_shape.reshape((68, 3))
        anchor_t_shape = anchor_t_shape[t_shape_idx, :]

    fl_data_i = fl_data_i.reshape((-1, 68, 3)).copy()

    # get rot_mat
    rot_quats = []
    rot_trans = []
    for i in range(fl_data_i.shape[0]):
        line = fl_data_i[i]
        frame_t_shape = line[t_shape_idx, :]
        T, distance, itr = icp(frame_t_shape, anchor_t_shape)
        rot_mat = T[:3, :3]
        trans_mat = T[:3, 3:4]

        # norm to anchor
        fl_data_i[i] = np.dot(rot_mat, line.T).T + trans_mat.T

        # inverse (anchor -> reat_t)
        # tmp = np.dot(rot_mat.T, (anchor_t_shape - trans_mat.T).T).T

        r = R.from_matrix(rot_mat)
        rot_quats.append(r.as_quat())
        # rot_eulers.append(r.as_euler('xyz'))
        rot_trans.append(T[:3, :])

    rot_quats = np.array(rot_quats)
    rot_trans = np.array(rot_trans)

    return rot_trans, rot_quats, fl_data_i

def close_face_lip(fl):
    facelandmark = fl.reshape(-1, 68, 3)
    from util.geo_math import area_of_polygon
    min_area_lip, idx = 999, 0
    for i, fls in enumerate(facelandmark):
        area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
        if (area_of_mouth < min_area_lip):
            min_area_lip = area_of_mouth
            idx = i
    return idx



class Speaker_aware_branch_Dataset(data.Dataset):

    def __init__(self, dump_dir, dump_name, num_window_frames, num_window_step, status, use_11spk_only=False, noautovc=''):
        self.dump_dir = dump_dir
        self.num_window_frames = num_window_frames
        self.num_window_step = num_window_step

        # Step 1 : load A / V data from dump files
        print('Loading Data {}_{}'.format(dump_name, status))

        with open(os.path.join(self.dump_dir, '{}_{}_{}au.pickle'.format(dump_name, status, noautovc)), 'rb') as fp:
            self.au_data = pickle.load(fp)
        with open(os.path.join(self.dump_dir, '{}_{}_{}fl.pickle'.format(dump_name, status, noautovc)), 'rb') as fp:
            self.fl_data = pickle.load(fp)
        try:
            with open(os.path.join(self.dump_dir, '{}_{}_gaze.pickle'.format(dump_name, status)), 'rb') as fp:
                gaze = pickle.load(fp)
                self.rot_trans = gaze['rot_trans']
                self.rot_quats = gaze['rot_quat']
                self.anchor_t_shape = gaze['anchor_t_shape']

                # print('raw:', np.sqrt(np.sum((logm(self.rot_trans[0][0, :3, :3].dot(self.rot_trans[0][5, :3, :3].T)))**2)/2.))
                # print('axis-angle:',np.arccos((np.sum(np.trace(self.rot_trans[0][0, :3, :3].dot(self.rot_trans[0][5, :3, :3].T)))-1.)/2.))
                # print('quat:', 2 * np.arccos(np.abs(self.rot_eulers[0][0].dot(self.rot_eulers[0][5].T))))
                # exit(0)
        except:
            print(os.path.join(self.dump_dir, '{}_{}_gaze.pickle'.format(dump_name, status)))
            print('gaze file not found')
            exit(-1)


        valid_idx = []
        for i, fl in enumerate(self.fl_data):
            if(use_11spk_only):
                if(fl[1][1][:-4].split('_x_')[1] in ['48uYS3bHIA8', 'E0zgrhQ0QDw', 'E_kmpT-EfOg', 'J-NPsvtQ8lE', 'Z7WRt--g-h4', '_ldiVrXgZKc', 'irx71tYyI-Q', 'sxCbrYjBsGA', 'wAAMEC1OsRc', 'W6uRNCJmdtI', 'bXpavyiCu10']):
                    # print(i, fl[1][1][:-4])
                    valid_idx.append(i)
            else:
                valid_idx.append(i)

        random.seed(0)
        random.shuffle(valid_idx)
        self.fl_data = [self.fl_data[i] for i in valid_idx]
        self.au_data = [self.au_data[i] for i in valid_idx]
        self.rot_trans = [self.rot_trans[i] for i in valid_idx]
        self.rot_quats = [self.rot_quats[i] for i in valid_idx]
        self.anchor_t_shape = [self.anchor_t_shape[i] for i in valid_idx]

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)

        # ''' PRODUCE gaze file for the first time '''
        # self.rot_trans = []
        # self.rot_quats = []
        # self.anchor_t_shape = []
        #
        # for fl in tqdm(self.fl_data):
        #     fl = fl[0].reshape((-1, 68, 3))
        #     rot_trans, rot_quats, anchor_t_shape = norm_output_fls_rot(fl, anchor_t_shape=None)
        #     self.rot_trans.append(rot_trans)
        #     self.rot_quats.append(rot_quats)
        #     self.anchor_t_shape.append(anchor_t_shape)
        #
        # with open(os.path.join(self.dump_dir, '{}_{}_gaze.pickle'.format(dump_name, status)), 'wb') as fp:
        #     gaze = {'rot_trans':self.rot_trans, 'rot_quat':self.rot_quats, 'anchor_t_shape':self.anchor_t_shape}
        #     pickle.dump(gaze, fp)
        #     print('SAVE!')


        au_mean_std = np.loadtxt('dataset/utils/MEAN_STD_AUTOVC_RETRAIN_MEL_AU.txt') # np.mean(self.au_data[0][0]), np.std(self.au_data[0][0])
        au_mean, au_std = au_mean_std[0:au_mean_std.shape[0]//2], au_mean_std[au_mean_std.shape[0]//2:]

        self.au_data = [((au - au_mean) / au_std, info) for au, info in self.au_data]

    def __len__(self):
        return  len(self.fl_data)

    def __getitem__(self, item):
        # print('-> get item {}: {} {}'.format(item, self.fl_data[item][1][0], self.fl_data[item][1][1]))
        return self.fl_data[item], self.au_data[item], self.rot_trans[item], \
                    self.rot_quats[item], self.anchor_t_shape[item]

    def my_collate_in_segments(self, batch):
        fls, aus, embs, regist_fls, rot_trans, rot_quats = [], [], [], [], [], []
        for fl, au, rot_tran, rot_quat, anchor_t_shape in batch:
            fl_data, au_data, emb_data = fl[0], au[0], au[1][2]
            assert (fl_data.shape[0] == au_data.shape[0])

            fl_data = torch.tensor(fl_data, dtype=torch.float, requires_grad=False)
            au_data = torch.tensor(au_data, dtype=torch.float, requires_grad=False)
            emb_data = torch.tensor(emb_data, dtype=torch.float, requires_grad=False)

            rot_tran_data = torch.tensor(rot_tran, dtype=torch.float, requires_grad=False)
            minus_eye = torch.cat([torch.eye(3).unsqueeze(0), torch.zeros((1, 3, 1))], dim=2)
            rot_tran_data -= minus_eye
            rot_quat_data = torch.tensor(rot_quat, dtype=torch.float, requires_grad=False)
            regist_fl_data = torch.tensor(anchor_t_shape, dtype=torch.float, requires_grad=False).view(-1, 204)

            # window shift data
            fls += [fl_data[i:i + self.num_window_frames] #- fl_data[i]
                    for i in range(0, fl_data.shape[0] - self.num_window_frames, self.num_window_step)]
            aus += [au_data[i:i + self.num_window_frames]
                    for i in range(0, au_data.shape[0] - self.num_window_frames, self.num_window_step)]
            embs += [emb_data] * ((au_data.shape[0] - self.num_window_frames) // self.num_window_step)

            regist_fls += [regist_fl_data[i:i + self.num_window_frames]  # - fl_data[i]
                    for i in range(0, regist_fl_data.shape[0] - self.num_window_frames, self.num_window_step)]
            rot_trans += [rot_tran_data[i:i + self.num_window_frames]  # - fl_data[i]
                    for i in range(0, rot_tran_data.shape[0] - self.num_window_frames, self.num_window_step)]
            rot_quats += [rot_quat_data[i:i + self.num_window_frames]  # - fl_data[i]
                    for i in range(0, rot_quat_data.shape[0] - self.num_window_frames, self.num_window_step)]

        fls = torch.stack(fls, dim=0)
        aus = torch.stack(aus, dim=0)
        embs = torch.stack(embs, dim=0)

        regist_fls = torch.stack(regist_fls, dim=0)
        rot_trans = torch.stack(rot_trans, dim=0)
        rot_quats = torch.stack(rot_quats, dim=0)

        return fls, aus, embs, regist_fls, rot_trans, rot_quats
