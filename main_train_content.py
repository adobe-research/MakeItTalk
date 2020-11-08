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
from src.approaches.train_content import Audio2landmark_model


ROOT_DIR = r'/mnt/ntfs/Dataset/TalkingToon/Obama_for_train'
DEMO_CH = ''

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default=ROOT_DIR, help='Root dir for data')
parser.add_argument('--nepoch', type=int, default=1001, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--in_batch_nepoch', type=int, default=1, help='')
parser.add_argument('--first_in_batch_nepoch', type=int, default=1, help='')
parser.add_argument('--segment_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_window_frames', type=int, default=18, help='')
parser.add_argument('--num_window_step', type=int, default=1, help='')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--dump_file_name', type=str, default='celeb_withrot', help='')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=0., help='weight decay')
parser.add_argument('--drop_out', type=float, default=0.5, help='drop out')
parser.add_argument('--verbose', type=int, default=1, help='0 - detail, 2 - simplify')
parser.add_argument('--write', default=False, action='store_true')

parser.add_argument('--add_pos', default=False, action='store_true')
parser.add_argument('--use_motion_loss', default=False, action='store_true')


parser.add_argument('--name', type=str, default='tmp')
parser.add_argument('--puppet_name', type=str, default=DEMO_CH)

parser.add_argument('--in_size', type=int, default=80)

parser.add_argument('--use_lip_weight', default=True, action='store_false')
parser.add_argument('--lambda_mse_loss', default=1.0, type=float)
parser.add_argument('--show_animation', default=False, action='store_true')

# model
parser.add_argument('--use_prior_net', default=True, action='store_false')
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--load_a2l_C_name', type=str, default='')
# arch
parser.add_argument('--use_reg_as_std', default=True, action='store_false')
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)

# test
parser.add_argument('--test_emb', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--test_end2end', default=False, action='store_true')

# save model
parser.add_argument('--jpg_freq', type=int, default=1, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=1, help='')
parser.add_argument('--random_clip_num', type=int, default=2, help='')


opt_parser = parser.parse_args()

root_dir = ROOT_DIR
# opt_parser.root_dir = ROOT_DIR
opt_parser.dump_dir = os.path.join(opt_parser.root_dir, 'dump')
opt_parser.ckpt_dir = os.path.join(opt_parser.root_dir, 'ckpt', opt_parser.name)
try_mkdir(opt_parser.ckpt_dir)
opt_parser.log_dir = os.path.join(opt_parser.root_dir, 'log')

# make directory for nn outputs
try_mkdir(opt_parser.dump_dir.replace('dump','nn_result'))
try_mkdir(os.path.join(opt_parser.dump_dir.replace('dump', 'nn_result'), opt_parser.name))


model = Audio2landmark_model(opt_parser)
if(opt_parser.train):
    model.train()
else:
    model.test()
