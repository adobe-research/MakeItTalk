"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import time
from src.dataset.audio2landmark.audio2landmark_dataset import Audio2landmark_Dataset
from src.models.model_audio2landmark import Audio2landmark_content
from util.utils import Record
from util.icp import icp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Audio2landmark_model():

    def __init__(self, opt_parser, jpg_shape=None):
        '''
        Init model with opt_parser
        '''
        print('Run on device:', device)

        # Step 1 : load opt_parser
        self.opt_parser = opt_parser
        self.std_face_id = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt')
        if(jpg_shape is not None):
            self.std_face_id = jpg_shape
        self.std_face_id = self.std_face_id.reshape(1, 204)
        self.std_face_id = torch.tensor(self.std_face_id, requires_grad=False, dtype=torch.float).to(device)

        self.train_data = Audio2landmark_Dataset(dump_dir=opt_parser.dump_dir,
                                                dump_name='autovc_retrain_mel',
                                                status='train',
                                               num_window_frames=opt_parser.num_window_frames,
                                               num_window_step=opt_parser.num_window_step)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=opt_parser.batch_size,
                                                           shuffle=False, num_workers=0,
                                                           collate_fn=self.train_data.my_collate_in_segments_noemb)
        print('TRAIN num videos: {}'.format(len(self.train_data)))

        self.eval_data = Audio2landmark_Dataset(dump_dir=opt_parser.dump_dir,
                                                 dump_name='autovc_retrain_mel',
                                                 status='test',
                                                 num_window_frames=opt_parser.num_window_frames,
                                                 num_window_step=opt_parser.num_window_step)
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_data, batch_size=opt_parser.batch_size,
                                                            shuffle=False, num_workers=0,
                                                            collate_fn=self.eval_data.my_collate_in_segments_noemb)
        print('EVAL num videos: {}'.format(len(self.eval_data)))

        # Step 3: Load model
        self.C = Audio2landmark_content(num_window_frames=opt_parser.num_window_frames, hidden_size=opt_parser.hidden_size,
                                      in_size=opt_parser.in_size, use_prior_net=opt_parser.use_prior_net,
                                      bidirectional=False, drop_out=opt_parser.drop_out)

        if(opt_parser.load_a2l_C_name.split('/')[-1] != ''):
            ckpt = torch.load(opt_parser.load_a2l_C_name)
            self.C.load_state_dict(ckpt['model_g_face_id'])
            print('======== LOAD PRETRAINED CONTENT BRANCH MODEL {} ========='.format(opt_parser.load_a2l_C_name))
        self.C.to(device)

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        self.anchor_t_shape = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt')
        self.anchor_t_shape = self.anchor_t_shape[self.t_shape_idx, :]

        self.opt_C = optim.Adam(self.C.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

        self.loss_mse = torch.nn.MSELoss()

    def __train_content__(self, fls, aus, face_id, is_training=True):

        fls_gt = fls[:, 0, :].detach().clone().requires_grad_(False)

        if (face_id.shape[0] == 1):
            face_id = face_id.repeat(aus.shape[0], 1)
        face_id = face_id.requires_grad_(False)

        fl_dis_pred, _ = self.C(aus, face_id)

        ''' lip region weight '''
        w = torch.abs(fls[:, 0, 66 * 3 + 1] - fls[:, 0, 62 * 3 + 1])
        w = torch.tensor([1.0]).to(device) / (w * 4.0 + 0.1)
        w = w.unsqueeze(1)
        lip_region_w = torch.ones((fls.shape[0], 204)).to(device)
        lip_region_w[:, 48*3:] = torch.cat([w] * 60, dim=1)
        lip_region_w = lip_region_w.detach().clone().requires_grad_(False)

        if (self.opt_parser.use_lip_weight):
            # loss = torch.mean(torch.mean((fl_dis_pred + face_id - fls[:, 0, :]) ** 2, dim=1) * w)
            loss = torch.mean(torch.abs(fl_dis_pred +face_id[0:1].detach() - fls_gt) * lip_region_w)
        else:
            # loss = self.loss_mse(fl_dis_pred + face_id, fls[:, 0, :])
            loss = torch.nn.functional.l1_loss(fl_dis_pred+face_id[0:1].detach(), fls_gt)

        if (self.opt_parser.use_motion_loss):
            pred_motion = fl_dis_pred[:-1] - fl_dis_pred[1:]
            gt_motion = fls_gt[:-1] - fls_gt[1:]
            loss += torch.nn.functional.l1_loss(pred_motion, gt_motion)

        ''' use laplacian smooth loss '''
        if (self.opt_parser.lambda_laplacian_smooth_loss > 0.0):
            n1 = [1] + list(range(0, 16)) + [18] + list(range(17, 21)) + [23] + list(range(22, 26)) + \
                 [28] + list(range(27, 35)) + [41] + list(range(36, 41)) + [47] + list(range(42, 47)) + \
                 [59] + list(range(48, 59)) + [67] + list(range(60, 67))
            n2 = list(range(1, 17)) + [15] + list(range(18, 22)) + [20] + list(range(23, 27)) + [25] + \
                 list(range(28, 36)) + [34] + list(range(37, 42)) + [36] + list(range(43, 48)) + [42] + \
                 list(range(49, 60)) + [48] + list(range(61, 68)) + [60]
            V = (fl_dis_pred + face_id[0:1].detach()).view(-1, 68, 3)
            L_V = V - 0.5 * (V[:, n1, :] + V[:, n2, :])
            G = fls_gt.view(-1, 68, 3)
            L_G = G - 0.5 * (G[:, n1, :] + G[:, n2, :])
            loss_laplacian = torch.nn.functional.l1_loss(L_V, L_G)
            loss += loss_laplacian

        if(is_training):
            self.opt_C.zero_grad()
            loss.backward()
            self.opt_C.step()

        if(not is_training):
            # ''' CALIBRATION '''
            np_fl_dis_pred = fl_dis_pred.detach().cpu().numpy()
            K = int(np_fl_dis_pred.shape[0] * 0.5)
            for calib_i in range(204):
                min_k_idx = np.argpartition(np_fl_dis_pred[:, calib_i], K)
                m = np.mean(np_fl_dis_pred[min_k_idx[:K], calib_i])
                np_fl_dis_pred[:, calib_i] = np_fl_dis_pred[:, calib_i] - m
            fl_dis_pred = torch.tensor(np_fl_dis_pred, requires_grad=False).to(device)

        return fl_dis_pred, face_id[0:1, :], loss

    def __train_pass__(self, epoch, log_loss, is_training=True):
        st_epoch = time.time()

        # Step 1: init setup
        if(is_training):
            self.C.train()
            data = self.train_data
            dataloader = self.train_dataloader
            status = 'TRAIN'
        else:
            self.C.eval()
            data = self.eval_data
            dataloader = self.eval_dataloader
            status = 'EVAL'

        random_clip_index = np.random.permutation(len(dataloader))[0:self.opt_parser.random_clip_num]
        print('random visualize clip index', random_clip_index)

        # Step 2: train for each batch
        for i, batch in enumerate(dataloader):

            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]
            inputs_fl, inputs_au = batch
            inputs_fl_ori, inputs_au_ori = inputs_fl.to(device), inputs_au.to(device)

            std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
            seg_bs = 512

            ''' pick a most closed lip frame from entire clip data '''
            close_fl_list = inputs_fl_ori[::10, 0, :]
            idx = self.__close_face_lip__(close_fl_list.detach().cpu().numpy())
            input_face_id = close_fl_list[idx:idx + 1, :]

            ''' register face '''
            if (self.opt_parser.use_reg_as_std):
                landmarks = input_face_id.detach().cpu().numpy().reshape(68, 3)
                frame_t_shape = landmarks[self.t_shape_idx, :]
                T, distance, itr = icp(frame_t_shape, self.anchor_t_shape)
                landmarks = np.hstack((landmarks, np.ones((68, 1))))
                registered_landmarks = np.dot(T, landmarks.T).T
                input_face_id = torch.tensor(registered_landmarks[:, 0:3].reshape(1, 204), requires_grad=False,
                                             dtype=torch.float).to(device)

            for in_batch in range(self.opt_parser.in_batch_nepoch):

                std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []

                if (is_training):
                    rand_start = np.random.randint(0, inputs_fl_ori.shape[0] // 5, 1).reshape(-1)
                    inputs_fl = inputs_fl_ori[rand_start[0]:]
                    inputs_au = inputs_au_ori[rand_start[0]:]
                else:
                    inputs_fl = inputs_fl_ori
                    inputs_au = inputs_au_ori

                for j in range(0, inputs_fl.shape[0], seg_bs):

                    # Step 3.1: load segments
                    inputs_fl_segments = inputs_fl[j: j + seg_bs]
                    inputs_au_segments = inputs_au[j: j + seg_bs]
                    fl_std = inputs_fl_segments[:, 0, :].data.cpu().numpy()

                    if(inputs_fl_segments.shape[0] < 10):
                        continue

                    fl_dis_pred_pos, input_face_id, loss = \
                        self.__train_content__(inputs_fl_segments, inputs_au_segments, input_face_id, is_training)

                    fl_dis_pred_pos = (fl_dis_pred_pos + input_face_id).data.cpu().numpy()
                    ''' solve inverse lip '''
                    fl_dis_pred_pos = self.__solve_inverse_lip2__(fl_dis_pred_pos)

                    fls_pred_pos_list += [fl_dis_pred_pos.reshape((-1, 204))]
                    std_fls_list += [fl_std.reshape((-1, 204))]

                    for key in log_loss.keys():
                        if (key not in locals().keys()):
                            continue
                        if (type(locals()[key]) == float):
                            log_loss[key].add(locals()[key])
                        else:
                            log_loss[key].add(locals()[key].data.cpu().numpy())


                if (epoch % self.opt_parser.jpg_freq == 0 and (i in random_clip_index or in_batch % self.opt_parser.jpg_freq == 1)):
                    def save_fls_av(fake_fls_list, postfix='', ifsmooth=True):
                        fake_fls_np = np.concatenate(fake_fls_list)
                        filename = 'fake_fls_{}_{}_{}.txt'.format(epoch, video_name, postfix)
                        np.savetxt(
                            os.path.join(self.opt_parser.dump_dir, '../nn_result', self.opt_parser.name, filename),
                            fake_fls_np, fmt='%.6f')
                        audio_filename = '{:05d}_{}_audio.wav'.format(global_id, video_name)
                        from util.vis import Vis_old
                        Vis_old(run_name=self.opt_parser.name, pred_fl_filename=filename, audio_filename=audio_filename,
                                fps=62.5, av_name='e{:04d}_{}_{}'.format(epoch, in_batch, postfix),
                                postfix=postfix, root_dir=self.opt_parser.root_dir, ifsmooth=ifsmooth)

                    if (self.opt_parser.show_animation and not is_training):
                        print('show animation ....')
                        save_fls_av(fls_pred_pos_list, 'pred_{}'.format(i), ifsmooth=True)
                        save_fls_av(std_fls_list, 'std_{}'.format(i), ifsmooth=False)
                        from util.vis import Vis_comp
                        Vis_comp(run_name=self.opt_parser.name,
                                 pred1='fake_fls_{}_{}_{}.txt'.format(epoch, video_name, 'pred_{}'.format(i)),
                                 pred2='fake_fls_{}_{}_{}.txt'.format(epoch, video_name, 'std_{}'.format(i)),
                                 audio_filename='{:05d}_{}_audio.wav'.format(global_id, video_name),
                                fps=62.5, av_name='e{:04d}_{}_{}'.format(epoch, in_batch, 'comp_{}'.format(i)),
                                postfix='comp_{}'.format(i), root_dir=self.opt_parser.root_dir, ifsmooth=False)

                    self.__save_model__(save_type='last_inbatch', epoch=epoch)

                if (self.opt_parser.verbose <= 1):
                    print('{} Epoch: #{} batch #{}/{} inbatch #{}/{}'.format(
                        status, epoch, i, len(dataloader),
                    in_batch, self.opt_parser.in_batch_nepoch), end=': ')
                    for key in log_loss.keys():
                        print(key, '{:.5f}'.format(log_loss[key].per('batch')), end=', ')
                    print('')

        if (self.opt_parser.verbose <= 2):
            print('==========================================================')
            print('{} Epoch: #{}'.format(status, epoch), end=':')
            for key in log_loss.keys():
                print(key, '{:.4f}'.format(log_loss[key].per('epoch')), end=', ')
            print(
                'Epoch time usage: {:.2f} sec\n==========================================================\n'.format(
                    time.time() - st_epoch))
        self.__save_model__(save_type='last_epoch', epoch=epoch)
        if (epoch % self.opt_parser.ckpt_epoch_freq == 0):
            self.__save_model__(save_type='e_{}'.format(epoch), epoch=epoch)


    def __close_face_lip__(self, fl):
        facelandmark = fl.reshape(-1, 68, 3)
        from util.geo_math import area_of_polygon
        min_area_lip, idx = 999, 0
        for i, fls in enumerate(facelandmark):
            area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < min_area_lip):
                min_area_lip = area_of_mouth
                idx = i
        return idx

    def test(self):
        eval_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}
        with torch.no_grad():
            self.__train_pass__(0, eval_loss, is_training=False)

    def train(self):
        train_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}
        eval_loss = {key: Record(['epoch', 'batch']) for key in ['loss']}

        for epoch in range(self.opt_parser.nepoch):
            self.__train_pass__(epoch=epoch, log_loss=train_loss)

            with torch.no_grad():
                self.__train_pass__(epoch, eval_loss, is_training=False)


    def __solve_inverse_lip2__(self, fl_dis_pred_pos_numpy):
        for j in range(fl_dis_pred_pos_numpy.shape[0]):
            init_face = self.std_face_id.detach().cpu().numpy()
            from util.geo_math import area_of_signed_polygon
            fls = fl_dis_pred_pos_numpy[j].reshape(68, 3)
            area_of_mouth = area_of_signed_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < 0):
                fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] + fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3])
                fl_dis_pred_pos_numpy[j, 63 * 3:64 * 3] = fl_dis_pred_pos_numpy[j, 65 * 3:66 * 3]
                fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] + fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3])
                fl_dis_pred_pos_numpy[j, 62 * 3:63 * 3] = fl_dis_pred_pos_numpy[j, 66 * 3:67 * 3]
                fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3] = 0.5 *(fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] + fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3])
                fl_dis_pred_pos_numpy[j, 61 * 3:62 * 3] = fl_dis_pred_pos_numpy[j, 67 * 3:68 * 3]
                p = max([j-1, 0])
                fl_dis_pred_pos_numpy[j, 55 * 3+1:59 * 3+1:3] = fl_dis_pred_pos_numpy[j, 64 * 3+1:68 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 55 * 3+1:59 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 64 * 3+1:68 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 59 * 3+1:60 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:61 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 59 * 3+1:60 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 60 * 3+1:61 * 3+1:3]
                fl_dis_pred_pos_numpy[j, 49 * 3+1:54 * 3+1:3] = fl_dis_pred_pos_numpy[j, 60 * 3+1:65 * 3+1:3] \
                                                          + fl_dis_pred_pos_numpy[p, 49 * 3+1:54 * 3+1:3] \
                                                          - fl_dis_pred_pos_numpy[p, 60 * 3+1:65 * 3+1:3]
        return fl_dis_pred_pos_numpy


    def __save_model__(self, save_type, epoch):
        if (self.opt_parser.write):
            torch.save({
                'model_g_face_id': self.C.state_dict(),
                'epoch': epoch
            }, os.path.join(self.opt_parser.ckpt_dir, 'ckpt_{}.pth'.format(save_type)))




