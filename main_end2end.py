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
from scipy.spatial.transform import Rotation as R
import time

DEBUG_MODE = False
ADD_NAIVE_EYE = True
GEN_AUDIO = True
GEN_FLS = True
head_name = 'paint_boy2'


parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='examples/{}.jpg'.format(head_name))


parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_a2l_db_e_875.pth') #ckpt_audio2landmark_g.pth') #
# parser.add_argument('--load_a2l_G_name', type=str, default='/mnt/ntfs/Dataset/TalkingToon/VoxCeleb2/ckpt/local_da_merge_3/ckpt_e_50.pth') #  local_r7_iter_all_1/ckpt_e_100.pth')
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_audio2landmark_c.pth')
# parser.add_argument('--load_a2l_C_name', type=str, default='/mnt/ntfs/Dataset/TalkingToon/Obama_for_train/ckpt/local_obama_norm_4_batch2/ckpt_e_100.pth')

parser.add_argument('--load_G_name', type=str, default='examples/ckpt/ckpt_116_i2i_comb.pth') #ckpt_image2image.pth') #                     ckpt_i2i_finetune_150.pth') #

parser.add_argument('--amp_lip_x', type=float, default=2.3)
parser.add_argument('--amp_lip_y', type=float, default=2.3)
parser.add_argument('--amp_pos', type=float, default=1.25)


parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

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
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')




opt_parser = parser.parse_args()

''' STEP 1: preprocess input single image '''
img =cv2.imread(opt_parser.jpg)
predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
shapes = predictor.get_landmarks(img)
if (not shapes or len(shapes) != 1):
    print('Cannot detect face landmarks. Exit.')
    exit(-1)
shape_3d = shapes[0]
shape_3d = np.concatenate([shape_3d, np.ones(shape=(68, 1))], axis=1)

# # close mouth
# shape_3d = shape_3d.reshape((1, 68, 3))
# index1 = list(range(60-1, 55-1, -1))
# index2 = list(range(68-1, 65-1, -1))
# mean_out = 0.5 * (shape_3d[:, 49:54] + shape_3d[:, index1])
# mean_in = 0.5 * (shape_3d[:, 61:64] + shape_3d[:, index2])
# shape_3d[:, 50:53] -= (shape_3d[:, 61:64] - mean_in) * 0.7
# shape_3d[:, list(range(59-1, 56-1, -1))] -= (shape_3d[:, index2] - mean_in) * 0.7
# shape_3d[:, 49] -= (shape_3d[:, 61] - mean_in[:, 0]) * 0.5
# shape_3d[:, 53] -= (shape_3d[:, 63] - mean_in[:, -1]) * 0.5
# shape_3d[:, 59] -= (shape_3d[:, 67] - mean_in[:, 0]) * 0.5
# shape_3d[:, 55] -= (shape_3d[:, 65] - mean_in[:, -1]) * 0.5
# # shape_3d[:, 61:64] = shape_3d[:, index2] = mean_in
# shape_3d[:, 61:64]  -= (shape_3d[:, 61:64] - mean_in) * 0.7
# shape_3d[:, index2] -= (shape_3d[:, index2] - mean_in) * 0.7
# shape_3d = shape_3d.reshape((68, 3))

if (DEBUG_MODE and True):
    print(shape_3d)
    while(True):
        from dataset.image_translation.data_preparation import vis_landmark_on_img
        vis_landmark_on_img(img, shape_3d.astype(np.int))
        cv2.imshow('jpg', img)
        c = cv2.waitKey(5)
        if(c == ord('q')):
            break

# dali
# shape_3d[49:54, 1] -= 1.
shape_3d[55:60, 1] -= 1.
# shape_3d[48:, 0] = (shape_3d[48:, 0] - np.mean(shape_3d[48:, 0])) * 1.1 + np.mean(shape_3d[48:, 0])
# shape_3d[[37,38,43,44], 1] -=2
# shape_3d[[40,41,46,47], 1] +=2


''' STEP 2: normalize face as input to audio branch '''
scale = 1.6 / (shape_3d[0, 0] - shape_3d[16, 0])
shift = - 0.5 * (shape_3d[0, 0:2] + shape_3d[16, 0:2])
shape_3d[:, 0:2] = (shape_3d[:, 0:2] + shift) * scale
face_std = np.loadtxt('dataset/utils/STD_FACE_LANDMARKS.txt').reshape(68, 3)
shape_3d[:, -1] = face_std[:, -1]
shape_3d[:, 0:2] = -shape_3d[:, 0:2]
print(shape_3d)

''' STEP 3: Generate audio data as input to audio branch '''
if(GEN_AUDIO):
    st = time.time()
    au_data = []
    au_emb = []
    ains = glob.glob1('examples', '*.wav')
    ains.sort()
    for ain in ains:
        os.system('ffmpeg -y -loglevel error -i examples/{} -ar 16000 examples/tmp.wav'.format(ain))
        shutil.copyfile('examples/tmp.wav', 'examples/{}'.format(ain))

        # au embedding
        from resemblyer_util.speaker_emb import get_spk_emb
        me, ae = get_spk_emb('examples/{}'.format(ain))
        au_emb.append(me.reshape(-1))


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

    print('TIME1', time.time() - st)


''' STEP 4: RUN audio->landmark network'''
if(GEN_FLS):
    from approaches.train_audio2landmark import Audio2landmark_model
    model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)
    import time
    st = time.time()
    model.test(au_emb=au_emb[0])
    print('TIME2', time.time() - st)

    #
    # from approaches.train_speaker_aware import Speaker_aware_branch
    # model = Speaker_aware_branch(opt_parser)
    # model.test_end2end(jpg_shape=shape_3d)

    print('finish gen fls')

''' STEP 5: de-normalize the output to the original image scale '''
fls = glob.glob1('examples', 'pred_fls_*.txt')
fls.sort()

for i in range(0,len(fls)):
    st = time.time()
    fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
    fl[:, :, 0:2] = -fl[:, :, 0:2]
    fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

    # fl[:, 0:8, 2] = fl[:, 8:9, 2]
    # fl[:, :, 2] = fl[:, :, 2] / scale * 0.5

    # old pose revise : local_r7_iter_all_1
    # fl[:, :, 0] += -40.
    # fl[:, :, 1] += -30.
    # fl[:, 48:, 1] += 2.
    # fl *= 0.95
    # fl[:, 48:, 1] += 2.

    # m = np.mean(fl[:, 48:, 1], axis=1, keepdims=True)
    # m[m < np.mean(fl[0, 48:, 1])] = np.mean(fl[0, 48:, 1])
    # fl[:, 48:, 1] = fl[:, 48:, 1] - np.mean(fl[:, 48:, 1], axis=1, keepdims=True) + m


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
            t += 60
            t += np.random.randint(30, 90)
            if (t < length - 1 - K2):
                close_time_stamp.append(t)
        for t in close_time_stamp:
            fl[t, 37], fl[t, 41] = 0.25 * fl[t, 37] + 0.75 * fl[t, 41], 0.25 * fl[t, 37] + 0.75 * fl[t, 41]
            fl[t, 38], fl[t, 40] = 0.25 * fl[t, 38] + 0.75 * fl[t, 40], 0.25 * fl[t, 38] + 0.75 * fl[t, 40]
            fl[t, 43], fl[t, 47] = 0.25 * fl[t, 43] + 0.75 * fl[t, 47], 0.25 * fl[t, 43] + 0.75 * fl[t, 47]
            fl[t, 44], fl[t, 46] = 0.25 * fl[t, 44] + 0.75 * fl[t, 46], 0.25 * fl[t, 44] + 0.75 * fl[t, 46]
            def interp_fl(t0, t1, t2, r):
                for index in [37, 38, 40, 41, 43, 44, 46, 47]:
                    fl[t0, index] = r * fl[t1, index] + (1-r) * fl[t2, index]
            for t0 in range(t-K1+1, t):
                interp_fl(t0, t-K1, t, r=(t-t0)/1./K1)
            for t0 in range(t+1, t+K2):
                interp_fl(t0, t, t+K2, r=(t+K2-1-t0)/1./K2)

    from scipy.signal import savgol_filter

    # fl = fl / (fl[:, 16, 0] - fl[:, 0, 0]).reshape((-1, 1, 1)) * (fl[0, 16, 0] - fl[0, 0, 0])

    fl = fl.reshape((-1, 204))
    fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
    fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
    fl = fl.reshape((-1, 68, 3))


    # ''' Auxiliar head pos '''
    # rot_fl = fl.reshape((-1, 3))
    # m = np.mean(rot_fl, axis=0, keepdims=True)
    # r = R.from_rotvec(-np.pi / 18. * 2.5 * np.array([0, 1, 0]))
    # T = r.as_matrix()
    # registered_landmarks = np.dot(T, (rot_fl-m).T).T
    # # r = R.from_rotvec(-np.pi / 18. * 1. * np.array([0, 0, 1]))
    # # T = r.as_matrix()
    # # registered_landmarks = np.dot(T, registered_landmarks.T).T
    # fl = (registered_landmarks + m).reshape((-1, 68, 3))
    # ''' ========================================     '''

    if (DEBUG_MODE and False):
        print(fl.shape)
        from dataset.image_translation.data_preparation import vis_landmark_on_img
        while(True):
            for item in fl:
                draw = img.copy()
                vis_landmark_on_img(draw, item.astype(np.int))
                cv2.imshow('jpg', draw)
                c = cv2.waitKey(10)
                if(c == ord('q')):
                    exit(0)

    ''' STEP 6: Imag2image translation '''
    model = Image_translation_block(opt_parser, single_test=True)
    with torch.no_grad():
        model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=head_name)
        print('finish image2image gen')

    print('TIME3', time.time() - st)
    print('frames', fl.shape)
