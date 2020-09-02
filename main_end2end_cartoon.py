"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('/home/yangzhou/Documents/Git/MakeItTalk_remote_copy/thirdparty/AdaptiveWingLoss')
import os, glob
import numpy as np
import cv2
import argparse
from dataset.image_translation.data_preparation import landmark_extraction, landmark_image_to_data
from approaches.train_image_translation import Image_translation_block
import platform
import torch
import pickle
import face_alignment
from autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
import shutil
from util.icp import icp

DEBUG_MODE = False
ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='examples/vdub2.jpg')

shape_3d = np.loadtxt('examples_cartoon/napkin_face_close_mouth.txt')
DEMO_CH = 'napkin'


parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_a2l_db_e_875.pth') #ckpt_audio2landmark_g.pth') #
# parser.add_argument('--load_a2l_G_name', type=str, default='/mnt/ntfs/Dataset/TalkingToon/VoxCeleb2/ckpt/local_da_merge_3/ckpt_e_50.pth') #  local_r7_iter_all_1/ckpt_e_100.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_audio2landmark_c.pth')
# parser.add_argument('--load_a2l_C_name', type=str, default='/mnt/ntfs/Dataset/TalkingToon/Obama_for_train/ckpt/local_obama_norm_4_batch2/ckpt_e_100.pth')

parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_i2i_finetune_150.pth') #ckpt_image2image.pth') #

parser.add_argument('--amp_lip_x', type=float, default=4.0)
parser.add_argument('--amp_lip_y', type=float, default=2.5)
parser.add_argument('--amp_pos', type=float, default=0.3)


parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples_cartoon')

#### NEW POSE MODEL
parser.add_argument('--test_end2end', default=True, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=512, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')


opt_parser = parser.parse_args()


''' STEP 3: Generate audio data as input to audio branch '''
if(GEN_AUDIO):
    au_data = []
    ains = glob.glob1('examples', '*.wav')
    ains.sort()
    for ain in ains:
        os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
        shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))
        print('Processing audio file', ain)
        c = AutoVC_mel_Convertor('examples')
        au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
               autovc_model_path=opt_parser.load_AUTOVC_name)
        au_data += au_data_i
        # os.remove(os.path.join('examples', 'tmp.wav'))
    if(os.path.isfile('examples/tmp.wav')):
        os.remove('examples/tmp.wav')

    fl_data = []
    rot_tran, rot_quat, anchor_t_shape = [], [], []
    for au, info in au_data:
        au_length = au.shape[0]
        fl = np.zeros(shape=(au_length, 68 * 3))
        fl_data.append((fl, info))
        rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
        rot_quat.append(np.zeros(shape=(au_length, 4)))
        anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

    if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
    if(os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))
    if(os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

    with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
        pickle.dump(fl_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
        pickle.dump(au_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
        gaze = {'rot_trans':rot_tran, 'rot_quat':rot_quat, 'anchor_t_shape':anchor_t_shape}
        pickle.dump(gaze, fp)


''' STEP 4: RUN audio->landmark network'''
if(GEN_FLS):
    from approaches.train_audio2landmark import Audio2landmark_model
    model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
    model.test()
    #
    # from approaches.train_speaker_aware import Speaker_aware_branch
    # model = Speaker_aware_branch(opt_parser)
    # model.test_end2end(jpg_shape=shape_3d)

    print('finish gen fls')

''' STEP 5: de-normalize the output to the original image scale '''
fls_names = glob.glob1('examples_cartoon', 'pred_fls_*.txt')
fls_names.sort()


for i in range(0,len(fls_names)):
    ains = glob.glob1('examples', '*.wav')
    ains.sort()
    ain = ains[i]
    fl = np.loadtxt(os.path.join('examples_cartoon', fls_names[i])).reshape((-1, 68,3))
    output_dir = os.path.join('examples_cartoon', fls_names[i][:-4])
    try:
        os.makedirs(output_dir)
    except:
        pass

    if (ADD_NAIVE_EYE):

        for t in range(fl.shape[0]):
            r = 0.95
            fl[t, 37], fl[t, 41] = r * fl[t, 37] + (1-r) * fl[t, 41], (1-r) * fl[t, 37] + r * fl[t, 41]
            fl[t, 38], fl[t, 40] = r * fl[t, 38] + (1-r) * fl[t, 40], (1-r) * fl[t, 38] + r * fl[t, 40]
            fl[t, 43], fl[t, 47] = r * fl[t, 43] + (1-r) * fl[t, 47], (1-r) * fl[t, 43] + r * fl[t, 47]
            fl[t, 44], fl[t, 46] = r * fl[t, 44] + (1-r) * fl[t, 46], (1-r) * fl[t, 44] + r * fl[t, 46]

        K1, K2 = 10, 15
        length = fl.shape[0]
        close_time_stamp = [30]
        t = 30
        while (t < length - 1 - K2):
            t += 90
            t += np.random.randint(0, 60)
            if (t < length - 1 - K2):
                close_time_stamp.append(t)
        for t in close_time_stamp:
            fl[t, 37], fl[t, 41] = 0.5 * (fl[t, 37] + fl[t, 41]), 0.5 * (fl[t, 37] + fl[t, 41])
            fl[t, 38], fl[t, 40] = 0.5 * (fl[t, 38] + fl[t, 40]), 0.5 * (fl[t, 38] + fl[t, 40])
            fl[t, 43], fl[t, 47] = 0.5 * (fl[t, 43] + fl[t, 47]), 0.5 * (fl[t, 43] + fl[t, 47])
            fl[t, 44], fl[t, 46] = 0.5 * (fl[t, 44] + fl[t, 46]), 0.5 * (fl[t, 44] + fl[t, 46])
            def interp_fl(t0, t1, t2, r):
                for index in [37, 38, 40, 41, 43, 44, 46, 47]:
                    fl[t0, index] = r * fl[t1, index] + (1-r) * fl[t2, index]
            for t0 in range(t-K1+1, t):
                interp_fl(t0, t-K1, t, r=(t-t0)/1./K1)
            for t0 in range(t+1, t+K2):
                interp_fl(t0, t, t+K2, r=(t+K2-1-t0)/1./K2)

    from util.utils import get_puppet_info

    bound, scale, shift = get_puppet_info(DEMO_CH, ROOT_DIR='examples_cartoon')

    fls = fl.reshape((-1, 68, 3))

    if(DEMO_CH == 'roy'):
        fls[:, 49:54, 1] += -0.02
        fls[:, 55:60, 1] -= -0.03

    fls[:, :, 0:2] = -fls[:, :, 0:2]
    fls[:, :, 0:2] = (fls[:, :, 0:2] / scale)
    fls[:, :, 0:2] -= shift.reshape(1, 2)

    fls = fls.reshape(-1, 204)

    # # predict_landmarks.txt
    from scipy.signal import savgol_filter
    fls[:, 0:48*3] = savgol_filter(fls[:, 0:48*3], 17, 3, axis=0)
    fls[:, 48*3:] = savgol_filter(fls[:, 48*3:], 5, 3, axis=0)

    fls = fls.reshape((-1, 68, 3))

    if DEMO_CH in ['wilk']:
        # ''' revise jaw distortion & bg anchor '''
        std_face = np.loadtxt(os.path.join('examples_cartoon', '{}_face_open_mouth.txt'.format(DEMO_CH)))
        std_face_jaw = std_face.reshape((-1, 68, 3))[:, 0:17, 0:2]
        jaw_l_reference = np.sqrt(
            np.sum((0.5 * (std_face_jaw[:, 0, :] + std_face_jaw[:, 16, :]) - std_face_jaw[:, 8, :]) ** 2, axis=1))
        fls_jaw = fls.reshape((-1, 68, 3))[:, 0:17, 0:2]
        jaw_l = np.sqrt(
            np.sum((0.5 * (fls_jaw[:, 0, :] + fls_jaw[:, 16, :]) - fls_jaw[:, 8, :]) ** 2, axis=1, keepdims=True))

        scaled_face_jaw = np.tile(std_face_jaw, (fls_jaw.shape[0], 1, 1))
        scaled_face_jaw[:, :, 1] *= (jaw_l / jaw_l_reference-1.) * 0.4 + 1.

        for i in range(scaled_face_jaw.shape[0]):
            src = scaled_face_jaw[i]
            trg = fls_jaw[i]
            T, distance, itr = icp(src, trg)
            rot_mat = T[:2, :2]
            scaled_face_jaw[i] = (np.dot(rot_mat, src.T) + T[:2, 2:3]).T
        # fls[:, 17:, 0:2] *= 0.99
        trg = fls.reshape((-1, 68, 3))
        src = std_face.reshape((-1, 68, 3))
        dis = - 1.0 * (0.5 * (scaled_face_jaw[:, 0, :] + scaled_face_jaw[:, 16, :]) - 0.5 * (
                    trg[:, 36, 0:2] + trg[:, 45, 0:2])) + \
              1.0 * (0.5 * (std_face_jaw[:, 0, :] + std_face_jaw[:, 16, :]) - 0.5 * (src[:, 36, 0:2] + src[:, 45, 0:2]))
        scaled_face_jaw = scaled_face_jaw + np.expand_dims(dis, axis=1)
        fls[:, 0:17, 0:2] = scaled_face_jaw
        fls[:, 0:17, 1] *= 1.02


    if (DEMO_CH in ['paint', 'mulaney', 'cartoonM', 'beer', 'color', 'JohnMulaney', 'vangogh', 'jm', 'roy', 'lineface']):
        r = list(range(0, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 160)

    else:
        r = list(range(0, 48)) + list(range(60, 68))
        fls = fls[:, r, :]
        fls = fls[:, :, 0:2].reshape(-1, 56 * 2)
        print(fls.shape, bound.shape)
        fls = np.concatenate((fls, np.tile(bound, (fls.shape[0], 1))), axis=1)
        fls = fls.reshape(-1, 112 + bound.shape[1])

    np.savetxt(os.path.join(output_dir, 'warped_points.txt'), fls, fmt='%.2f')

    # static_points.txt
    static_frame = np.loadtxt(os.path.join('examples_cartoon', '{}_face_open_mouth.txt'.format(DEMO_CH)))
    static_frame = static_frame[r, 0:2]
    static_frame = np.concatenate((static_frame, bound.reshape(-1, 2)), axis=0)
    np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame, fmt='%.2f')

    # triangle_vtx_index.txt
    shutil.copy(os.path.join('examples_cartoon', DEMO_CH + '_delauney_tri.txt'),
                os.path.join(output_dir, 'triangulation.txt'))

    # # ==============================================
    # # Step 4 : Jukub's morphing
    # # ==============================================
    # warp_exe = os.path.join(os.getcwd(), 'facewarp', 'dingwarp.exe')
    # import os
    #
    # if (os.path.exists(os.path.join(output_dir, 'output'))):
    #     shutil.rmtree(os.path.join(output_dir, 'output'))
    # os.mkdir(os.path.join(output_dir, 'output'))
    # os.chdir('{}'.format(os.path.join(output_dir, 'output')))
    # print(os.getcwd())
    #
    # os.system('{} {} {} {} {} {}'.format(
    #     warp_exe,
    #     os.path.join('examples_cartoon', DEMO_CH+'.png'),
    #     os.path.join(output_dir, 'triangulation.txt'),
    #     os.path.join(output_dir, 'reference_points.txt'),
    #     os.path.join(output_dir, 'warped_points.txt'),
    #     # os.path.join(ROOT_DIR, 'puppets', sys.argv[6]),
    #     '-novsync -dump'))
    # os.system('ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i {} -shortest {}'.format(
    #     ain,
    #     os.path.join(output_dir, sys.argv[8])
    # ))

    # shutil.copy(os.path.join(nn_result_dir, sys.argv[8]), os.path.join(nn_result_dir, '../../demo_result', sys.argv[8]))


    # MACOS
    # WINARCH=win64 WINEPREFIX=~/.wine-64prefix wine ../dingwarp.exe ../onepunch.png ../triangulation.txt ../reference_points.txt ../warped_points.txt ../onepunch_onlybody.jpg -novsync -dump
    # ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i ../tim_02_16k.wav -shortest -pix_fmt yuv420p ../c02_7.mp4