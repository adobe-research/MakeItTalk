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
import torch.nn as nn
from src.dataset.audio2landmark import Audio2landmark_Dataset
from src.models import Audio2landmark_speaker_aware
from util.utils import Record, get_n_params
from tensorboardX import SummaryWriter
from util.icp import icp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Speaker_aware_branch():

    def __init__(self, opt_parser):
        print('Run on device:', device)

        # Step 1 : load opt_parser
        for key in vars(opt_parser).keys():
            print(key, ':', vars(opt_parser)[key])

        self.opt_parser = opt_parser
        self.dump_dir = opt_parser.dump_dir
        self.std_face_id = np.loadtxt('dataset/utils/STD_FACE_LANDMARKS.txt')
        self.std_face_id = self.std_face_id.reshape(1, 204)
        self.std_face_id = torch.tensor(self.std_face_id, requires_grad=False, dtype=torch.float).to(device)

        # Step 2 : load data
        self.train_data = Audio2landmark_Dataset(dump_dir=self.dump_dir, dump_name=opt_parser.dump_file_name,
                                                num_window_frames=opt_parser.num_window_frames,
                                                num_window_step=opt_parser.num_window_step,
                                                status='train', noautovc='noautovc_')
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=opt_parser.batch_size,
                                                            shuffle=False, num_workers=0,
                                                            collate_fn=self.train_data.my_collate_in_segments_noemb)

        print('Train num videos: {}'.format(len(self.train_data)))
        self.eval_data = Audio2landmark_Dataset(dump_dir=self.dump_dir, dump_name=opt_parser.dump_file_name,
                                               num_window_frames=opt_parser.num_window_frames,
                                               num_window_step=opt_parser.num_window_step,
                                               status='val', noautovc='noautovc_')
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_data, batch_size=opt_parser.batch_size,
                                                           shuffle=False, num_workers=0,
                                                           collate_fn=self.eval_data.my_collate_in_segments_noemb)
        print('EVAL num videos: {}'.format(len(self.eval_data)))

        # Step 3: Load model
        self.G = Audio2landmark_speaker_aware(
                                     spk_emb_enc_size=opt_parser.spk_emb_enc_size,
                                     transformer_d_model=opt_parser.transformer_d_model,
                                     N=opt_parser.transformer_N, heads=opt_parser.transformer_heads,
                                     pos_dim=opt_parser.pos_dim,
                                     use_prior_net=True, is_noautovc=True)
        # self.G.apply(weight_init)
        for p in self.G.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('G: Running on {}, total num params = {:.2f}M'.format(device, get_n_params(self.G)/1.0e6))

        # self.D_L = Audio2landmark_pos_DL()
        # self.D_L.apply(weight_init)
        # print('D_L: Running on {}, total num params = {:.2f}M'.format(device, get_n_params(self.D_L)/1.0e6))
        #
        # self.D_T = Audio2landmark_pos_DT(spk_emb_enc_size=opt_parser.spk_emb_enc_size,
        #                                      transformer_d_model=opt_parser.transformer_d_model,
        #                                      N=opt_parser.transformer_N, heads=opt_parser.transformer_heads)
        # for p in self.D_T.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # print('D_T: Running on {}, total num params = {:.2f}M'.format(device, get_n_params(self.D_T) / 1.0e6))

        if (opt_parser.load_a2l_G_name.split('/')[-1] != ''):
            model_dict = self.G.state_dict()
            ckpt = torch.load(opt_parser.load_a2l_G_name)
            pretrained_dict = {k: v for k, v in ckpt['G'].items()
                               if 'out.' not in k and 'out_pos_1.' not in k}
            model_dict.update(pretrained_dict)

            self.G.load_state_dict(model_dict)
            print('======== LOAD PRETRAINED SPEAKER AWARE MODEL {} ========='.format(opt_parser.load_a2l_G_name))
        self.G.to(device)

        self.loss_mse = torch.nn.MSELoss()
        self.loss_bce = torch.nn.BCELoss()

        self.opt_G = optim.Adam(self.G.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

        if (opt_parser.write):
            self.writer = SummaryWriter(log_dir=os.path.join(opt_parser.log_dir, opt_parser.name))
            self.writer_count = {'TRAIN_epoch': 0, 'TRAIN_batch': 0, 'TRAIN_in_batch': 0,
                                 'EVAL_epoch': 0, 'EVAL_batch': 0, 'EVAL_in_batch': 0}

        self.t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
        self.anchor_t_shape = np.loadtxt('dataset/utils//STD_FACE_LANDMARKS.txt')
        self.anchor_t_shape = self.anchor_t_shape[self.t_shape_idx, :]

    def __train_speaker_aware__(self, fls, aus, face_id, is_training=True):

        # fls_gt = fls[:, 0, :].detach().clone().requires_grad_(False)
        reg_fls_gt = fls[:, 0, :].detach().clone().requires_grad_(False)

        if (face_id.shape[0] == 1):
            face_id = face_id.repeat(aus.shape[0], 1)
        face_id = face_id.requires_grad_(False)
        content_branch_face_id = face_id.detach()

        ''' ======================================================
                        Generator  G
        ====================================================== '''

        for name, p in self.G.named_parameters():
            p.requires_grad = True

        fl_dis_pred, pos_pred, _, spk_encode = self.G(aus, face_id)

        # reg fls loss
        loss_reg_fls = torch.nn.functional.l1_loss(fl_dis_pred+face_id[0:1].detach(), reg_fls_gt)

        # reg fls laplacian
        ''' use laplacian smooth loss '''
        loss_laplacian = 0.
        if (self.opt_parser.lambda_laplacian_smooth_loss > 0.0):
            n1 = [1] + list(range(0, 16)) + [18] + list(range(17, 21)) + [23] + list(range(22, 26)) + \
                 [28] + list(range(27, 35)) + [41] + list(range(36, 41)) + [47] + list(range(42, 47)) + \
                 [59] + list(range(48, 59)) + [67] + list(range(60, 67))
            n2 = list(range(1, 17)) + [15] + list(range(18, 22)) + [20] + list(range(23, 27)) + [25] + \
                 list(range(28, 36)) + [34] + list(range(37, 42)) + [36] + list(range(43, 48)) + [42] + \
                 list(range(49, 60)) + [48] + list(range(61, 68)) + [60]
            V = (fl_dis_pred + face_id[0:1].detach()).view(-1, 68, 3)
            L_V = V - 0.5 * (V[:, n1, :] + V[:, n2, :])
            G = reg_fls_gt.view(-1, 68, 3)
            L_G = G - 0.5 * (G[:, n1, :] + G[:, n2, :])
            loss_laplacian = torch.nn.functional.l1_loss(L_V, L_G)

        loss = loss_reg_fls + loss_laplacian * self.opt_parser.lambda_laplacian_smooth_loss
        # loss = loss_pos

        if(is_training):
            self.opt_G.zero_grad()
            loss.backward()
            self.opt_G.step()

        # reconstruct face through pos
        fl_dis_pred = fl_dis_pred + face_id[0:1].detach()

        return fl_dis_pred, pos_pred, face_id[0:1, :], (loss, loss_reg_fls, loss_laplacian)

    def __train_pass__(self, epoch, log_loss, is_training=True):
        st_epoch = time.time()

        # Step 1: init setup
        if (is_training):
            self.G.train()
            data = self.train_data
            dataloader = self.train_dataloader
            status = 'TRAIN'
        else:
            self.G.eval()
            data = self.eval_data
            dataloader = self.eval_dataloader
            status = 'EVAL'

        # random_clip_index = np.random.randint(0, len(dataloader)-1, 4)
        # random_clip_index = np.random.randint(0, 64, 4)
        random_clip_index = list(range(len(dataloader)))
        # print('random_clip_index', random_clip_index)
        # Step 2: train for each batch
        for i, batch in enumerate(dataloader):

            # if(i>=512):
            #     break

            st = time.time()
            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]

            # Step 2.1: load batch data from dataloader (in segments)
            inputs_fl, inputs_au = batch

            if (is_training):
                rand_start = np.random.randint(0, inputs_fl.shape[0] // 5, 1).reshape(-1)
                inputs_fl = inputs_fl[rand_start[0]:]
                inputs_au = inputs_au[rand_start[0]:]

            inputs_fl, inputs_au = inputs_fl.to(device), inputs_au.to(device)
            std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
            seg_bs = self.opt_parser.segment_batch_size

            close_fl_list = inputs_fl[::10, 0, :]
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

            for j in range(0, inputs_fl.shape[0], seg_bs):
                # Step 3.1: load segments
                inputs_fl_segments = inputs_fl[j: j + seg_bs]
                inputs_au_segments = inputs_au[j: j + seg_bs]


                if(inputs_fl_segments.shape[0] < 10):
                    continue

                if(self.opt_parser.test_emb):
                    input_face_id = self.std_face_id

                fl_dis_pred_pos, pos_pred, input_face_id, (loss, loss_g, loss_laplacian) = \
                    self.__train_speaker_aware__(inputs_fl_segments, inputs_au_segments, input_face_id,
                                                 is_training=is_training)

                fl_dis_pred_pos = fl_dis_pred_pos.data.cpu().numpy()
                fl_std = inputs_fl_segments[:, 0, :].data.cpu().numpy()
                ''' solve inverse lip '''
                if(not is_training):
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

            if (epoch % 5 == 0): # and i in [0, 200, 400, 600, 800, 1000]):
                def save_fls_av(fake_fls_list, postfix='', ifsmooth=True):
                    fake_fls_np = np.concatenate(fake_fls_list)
                    filename = 'fake_fls_{}_{}_{}.txt'.format(epoch, video_name, postfix)
                    np.savetxt(
                        os.path.join(self.opt_parser.dump_dir, '../nn_result', self.opt_parser.name, filename),
                        fake_fls_np, fmt='%.6f')
                    # audio_filename = '{:05d}_{}_audio.wav'.format(global_id, video_name)
                    # from util.vis import Vis_old
                    # Vis_old(run_name=self.opt_parser.name, pred_fl_filename=filename, audio_filename=audio_filename,
                    #     fps=62.5, av_name='e{:04d}_{}_{}'.format(epoch, i, postfix),
                    #     postfix=postfix, root_dir=self.opt_parser.root_dir, ifsmooth=ifsmooth)

                if (True):
                    if (self.opt_parser.show_animation):
                        print('show animation ....')
                        save_fls_av(fls_pred_pos_list, 'pred', ifsmooth=True)
                        save_fls_av(std_fls_list, 'std', ifsmooth=False)

            if (self.opt_parser.verbose <= 1):
                print('{} Epoch: #{} batch #{}/{}'.format(status, epoch, i, len(dataloader)), end=': ')
                for key in log_loss.keys():
                    print(key, '{:.5f}'.format(log_loss[key].per('batch')), end=', ')
                print('')
            self.__tensorboard_write__(status, log_loss, 'batch')

        if (self.opt_parser.verbose <= 2):
            print('==========================================================')
            print('{} Epoch: #{}'.format(status, epoch), end=':')
            for key in log_loss.keys():
                print(key, '{:.4f}'.format(log_loss[key].per('epoch')), end=', ')
            print('Epoch time usage: {:.2f} sec\n==========================================================\n'.format(time.time() - st_epoch))
        self.__save_model__(save_type='last_epoch', epoch=epoch)
        if(epoch % 5 == 0):
            self.__save_model__(save_type='e_{}'.format(epoch), epoch=epoch)
        self.__tensorboard_write__(status, log_loss, 'epoch')


    def test_end2end(self, jpg_shape):

        self.G.eval()
        self.C.eval()
        data = self.eval_data
        dataloader = self.eval_dataloader

        for i, batch in enumerate(dataloader):

            global_id, video_name = data[i][0][1][0], data[i][0][1][1][:-4]

            inputs_fl, inputs_au, inputs_emb, inputs_reg_fl, inputs_rot_tran, inputs_rot_quat = batch

            for key in ['irx71tYyI-Q', 'J-NPsvtQ8lE', 'Z7WRt--g-h4', 'E0zgrhQ0QDw', 'bXpavyiCu10', 'W6uRNCJmdtI', 'sxCbrYjBsGA', 'wAAMEC1OsRc', '_ldiVrXgZKc', '48uYS3bHIA8', 'E_kmpT-EfOg']:
                emb_val = self.test_embs[key]
                inputs_emb = np.tile(emb_val, (inputs_emb.shape[0], 1))
                inputs_emb = torch.tensor(inputs_emb, dtype=torch.float, requires_grad=False)

                # this_emb = key
                # inputs_emb = torch.zeros(size=(inputs_au.shape[0], len(self.test_embs_dic.keys())))
                # inputs_emb[:, self.test_embs_dic[this_emb]] = 1.

                inputs_fl, inputs_au, inputs_emb = inputs_fl.to(device), inputs_au.to(device), inputs_emb.to(device)
                inputs_reg_fl, inputs_rot_tran, inputs_rot_quat = inputs_reg_fl.to(device), inputs_rot_tran.to(device), inputs_rot_quat.to(device)

                std_fls_list, fls_pred_face_id_list, fls_pred_pos_list = [], [], []
                seg_bs = self.opt_parser.segment_batch_size

                # input_face_id = self.std_face_id
                input_face_id = torch.tensor(jpg_shape.reshape(1, 204), requires_grad=False, dtype=torch.float).to(device)

                ''' register face '''
                if (True):
                    landmarks = input_face_id.detach().cpu().numpy().reshape(68, 3)
                    frame_t_shape = landmarks[self.t_shape_idx, :]
                    T, distance, itr = icp(frame_t_shape, self.anchor_t_shape)
                    landmarks = np.hstack((landmarks, np.ones((68, 1))))
                    registered_landmarks = np.dot(T, landmarks.T).T
                    input_face_id = torch.tensor(registered_landmarks[:, 0:3].reshape(1, 204), requires_grad=False,
                                                 dtype=torch.float).to(device)

                for j in range(0, inputs_fl.shape[0], seg_bs):
                    # Step 3.1: load segments
                    inputs_fl_segments = inputs_fl[j: j + seg_bs]
                    inputs_au_segments = inputs_au[j: j + seg_bs]
                    inputs_emb_segments = inputs_emb[j: j + seg_bs]
                    inputs_reg_fl_segments = inputs_reg_fl[j: j + seg_bs]
                    inputs_rot_tran_segments = inputs_rot_tran[j: j + seg_bs]
                    inputs_rot_quat_segments = inputs_rot_quat[j: j + seg_bs]

                    if(inputs_fl_segments.shape[0] < 10):
                        continue

                    fl_dis_pred_pos, pos_pred, input_face_id, (loss, loss_reg_fls, loss_laplacian, loss_pos) = \
                        self.__train_speaker_aware__(inputs_fl_segments, inputs_au_segments, inputs_emb_segments,
                                                       input_face_id,  inputs_reg_fl_segments, inputs_rot_tran_segments,
                                                     inputs_rot_quat_segments,
                                                     is_training=False, use_residual=True)

                    fl_dis_pred_pos = fl_dis_pred_pos.data.cpu().numpy()
                    pos_pred = pos_pred.data.cpu().numpy()
                    fl_std = inputs_reg_fl_segments[:, 0, :].data.cpu().numpy()
                    pos_std = inputs_rot_tran_segments[:, 0, :].data.cpu().numpy()

                    ''' solve inverse lip '''
                    fl_dis_pred_pos = self.__solve_inverse_lip2__(fl_dis_pred_pos)

                    fl_dis_pred_pos = fl_dis_pred_pos.reshape((-1, 68, 3))
                    fl_std = fl_std.reshape((-1, 68, 3))
                    if(self.opt_parser.pos_dim == 12):
                        pos_pred = pos_pred.reshape((-1, 3, 4))
                        for k in range(fl_dis_pred_pos.shape[0]):
                            fl_dis_pred_pos[k] = np.dot(pos_pred[k, :3, :3].T + np.eye(3),
                                                        (fl_dis_pred_pos[k] - pos_pred[k, :, 3].T).T).T
                        pos_std = pos_std.reshape((-1, 3, 4))
                        for k in range(fl_std.shape[0]):
                            fl_std[k] = np.dot(pos_std[k, :3, :3].T + np.eye(3),
                                                        (fl_std[k] - pos_std[k, :, 3].T).T).T
                    else:
                        smooth_length = int(min(pos_pred.shape[0] - 1, 27) // 2 * 2 + 1)
                        pos_pred = savgol_filter(pos_pred, smooth_length, 3, axis=0)
                        quat = pos_pred[:, :4]
                        trans = pos_pred[:, 4:]
                        for k in range(fl_dis_pred_pos.shape[0]):
                            fl_dis_pred_pos[k] = np.dot(R.from_quat(quat[k]).as_matrix().T,
                                                        (fl_dis_pred_pos[k] - trans[k:k+1]).T).T
                        pos_std = pos_std.reshape((-1, 3, 4))
                        for k in range(fl_std.shape[0]):
                            fl_std[k] = np.dot(pos_std[k, :3, :3].T + np.eye(3),
                                               (fl_std[k] - pos_std[k, :, 3].T).T).T

                    fls_pred_pos_list += [fl_dis_pred_pos.reshape((-1, 204))]
                    std_fls_list += [fl_std.reshape((-1, 204))]

                fake_fls_np = np.concatenate(fls_pred_pos_list)
                filename = 'pred_fls_{}_{}.txt'.format(video_name.split('/')[-1], key)
                np.savetxt(os.path.join('examples', filename), fake_fls_np, fmt='%.6f')


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

    def train(self):
        train_loss = {key: Record(['epoch', 'batch']) for key in
                    ['loss','loss_laplacian', 'loss_reg_fls', 'loss_pos']}

        eval_loss = {key: Record(['epoch', 'batch']) for key in
                     ['loss','loss_laplacian', 'loss_reg_fls', 'loss_pos']}
        for epoch in range(self.opt_parser.nepoch):
            self.__train_pass__(epoch=epoch, log_loss=train_loss, is_training=True)
            # with torch.no_grad():
            #     self.__train_pass__(epoch=epoch, log_loss=eval_loss, is_training=False)

    def test(self):
        train_loss = {key: Record(['epoch', 'batch', 'in_batch']) for key in
                      ['loss', 'loss_g', 'loss_laplacian']}
        eval_loss = {key: Record(['epoch', 'batch', 'in_batch']) for key in
                     ['loss_pos', 'loss_g', 'loss_laplacian']}
        with torch.no_grad():
            self.__train_pass__(epoch=0, log_loss=eval_loss, is_training=False)

    def __tensorboard_write__(self, status, loss, t):
        if (self.opt_parser.write):
            for key in loss.keys():
                self.writer.add_scalar('{}_loss_{}_{}'.format(status, t, key), loss[key].per(t),
                                       self.writer_count[status + '_' + t])
                loss[key].clean(t)
            self.writer_count[status + '_' + t] += 1
        else:
            for key in loss.keys():
                loss[key].clean(t)

    def __save_model__(self, save_type, epoch):
        if (self.opt_parser.write):
            torch.save({
                'G': self.G.state_dict(),
                'epoch': epoch
            }, os.path.join(self.opt_parser.ckpt_dir, 'ckpt_{}.pth'.format(save_type)))

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt_parser.lr * (0.3 ** (np.max((0, epoch + 0)) // 50))
        lr = np.max((lr, 1e-5))
        print('###### ==== > Adjust learning rate to ', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # print('lr:', param_group['lr'])

    def __solve_inverse_lip2__(self, fl_dis_pred_pos_numpy):
        for j in range(fl_dis_pred_pos_numpy.shape[0]):
            # init_face = self.std_face_id.detach().cpu().numpy()
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



