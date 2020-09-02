"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import os, glob
import numpy as np
import cv2
import argparse
import platform
import torch
from util.utils import try_mkdir
from approaches.train_speaker_aware import Speaker_aware_branch


if platform.release() == '4.4.0-83-generic':
    ROOT_DIR = r'/mnt/ntfs/Dataset/TalkingToon/VoxCeleb2'
else: # 3.10.0-957.21.2.el7.x86_64
    ROOT_DIR = r'/mnt/nfs/work1/kalo/yangzhou/TalkingToon/VoxCeleb2'

DEMO_CH = ''

parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=1001, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--in_batch_nepoch', type=int, default=1, help='')
parser.add_argument('--first_in_batch_nepoch', type=int, default=1, help='')
parser.add_argument('--segment_batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_window_frames', type=int, default=18, help='')
parser.add_argument('--num_window_frames_seq', type=int, default=18, help='')
parser.add_argument('--num_window_frames_sync', type=int, default=18, help='')
parser.add_argument('--num_window_step', type=int, default=1, help='')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--dump_file_name', type=str, default='celeb_normrot', help='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--drop_out', type=float, default=0, help='drop out')
parser.add_argument('--verbose', type=int, default=1, help='0 - detail, 2 - simplify')
parser.add_argument('--write', default=False, action='store_true')

parser.add_argument('--add_pos', default=False, action='store_true')
parser.add_argument('--use_motion_loss', default=False, action='store_true')


parser.add_argument('--name', type=str, default='tmp')
parser.add_argument('--puppet_name', type=str, default=DEMO_CH)

parser.add_argument('--in_size', type=int, default=80)

parser.add_argument('--use_lip_weight', default=False, action='store_true')
parser.add_argument('--use_adain', default=False, action='store_true')
parser.add_argument('--use_residual', default=False, action='store_true')
parser.add_argument('--use_norm_emb', default=False, action='store_true')
parser.add_argument('--use_cycle_loss', default=False, action='store_true')
parser.add_argument('--lambda_cycle_loss', default=1.0, type=float)
parser.add_argument('--emb_coef', default=3.0, type=float)

parser.add_argument('--freeze_content_emb', default=False, action='store_true')
parser.add_argument('--pretrain_g', default=False, action='store_true')

parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--c_enc_hidden_size', default=256, type=int)
parser.add_argument('--lstm_g_hidden_size', default=256, type=int)
parser.add_argument('--projection_size', default=512, type=int)

parser.add_argument('--use_addinfo_format', default='motion_and_pos')
parser.add_argument('--l2_on_fls_without_traj', default=False, action='store_true')
parser.add_argument('--train_with_grad_penalty', default=False, action='store_true')
parser.add_argument('--train_DL', default=-1.0, type=float)
parser.add_argument('--train_DT', default=-1.0, type=float)
parser.add_argument('--train_G_only', default=False, action='store_true')
parser.add_argument('--lambda_mse_loss', default=1.0, type=float)
parser.add_argument('--teacher_force', default=0.0, type=float)
parser.add_argument('--debug_version', default='', type=str)
parser.add_argument('--lambda_add_info_loss', default=1.0, type=float)


parser.add_argument('--show_animation', default=False, action='store_true')



# model
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--load_a2l_C_name', type=str, default='examples/ckpt/ckpt_audio2landmark_c.pth')
parser.add_argument('--init_content_encoder', type=str, default='examples/ckpt/ckpt_audio2landmark_c.pth') #  'tt_lipwpre_prior_useclose/ckpt_last_epoch_20.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='/mnt/ntfs/Dataset/TalkingToon/VoxCeleb2/ckpt/local_da_merge_3/ckpt_e_50.pth') #


# data
parser.add_argument('--use_11spk_only', default=True, action='store_true')

# arch
parser.add_argument('--use_reg_as_std', default=True, action='store_false')
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)

# test
parser.add_argument('--test_emb', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test_end2end', default=False, action='store_true')

# save model
parser.add_argument('--jpg_freq', type=int, default=25, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=25, help='')

AMP = {'default':[2.5, 2.5, 1.0]}
if(DEMO_CH not in AMP.keys()):
    AMP[DEMO_CH] = AMP['default']

parser.add_argument('--amp_lip_x', type=float, default=AMP[DEMO_CH][0])
parser.add_argument('--amp_lip_y', type=float, default=AMP[DEMO_CH][1])
parser.add_argument('--amp_pos', type=float, default=AMP[DEMO_CH][2])

opt_parser = parser.parse_args()

root_dir = ROOT_DIR
opt_parser.root_dir = ROOT_DIR
opt_parser.dump_dir = os.path.join(root_dir, 'dump')
opt_parser.ckpt_dir = os.path.join(root_dir, 'ckpt', opt_parser.name)
try_mkdir(opt_parser.ckpt_dir)
opt_parser.log_dir = os.path.join(root_dir, 'log')

# make directory for nn outputs
try_mkdir(opt_parser.dump_dir.replace('dump','nn_result'))
try_mkdir(os.path.join(opt_parser.dump_dir.replace('dump', 'nn_result'), opt_parser.name))


model = Speaker_aware_branch(opt_parser)
if(opt_parser.train):
    model.train()
else:
    model.test()