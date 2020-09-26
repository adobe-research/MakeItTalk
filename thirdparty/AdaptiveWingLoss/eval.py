from __future__ import print_function, division
import torch
import argparse
import numpy as np
import torch.nn as nn
import time
import os
from core.evaler import eval_model
from core.dataloader import get_dataset
from core import models
from tensorboardX import SummaryWriter

# Parse arguments
parser = argparse.ArgumentParser()
# Dataset paths
parser.add_argument('--val_img_dir', type=str,
                    help='Validation image directory')
parser.add_argument('--val_landmarks_dir', type=str,
                    help='Validation landmarks directory')
parser.add_argument('--num_landmarks', type=int, default=68,
                    help='Number of landmarks')

# Checkpoint and pretrained weights
parser.add_argument('--ckpt_save_path', type=str,
                    help='a directory to save checkpoint file')
parser.add_argument('--pretrained_weights', type=str,
                    help='a directory to save pretrained_weights')

# Eval options
parser.add_argument('--batch_size', type=int, default=25,
                    help='learning rate decay after each epoch')

# Network parameters
parser.add_argument('--hg_blocks', type=int, default=4,
                    help='Number of HG blocks to stack')
parser.add_argument('--gray_scale', type=str, default="False",
                    help='Whether to convert RGB image into gray scale during training')
parser.add_argument('--end_relu', type=str, default="False",
                    help='Whether to add relu at the end of each HG module')

args = parser.parse_args()

VAL_IMG_DIR = args.val_img_dir
VAL_LANDMARKS_DIR = args.val_landmarks_dir
CKPT_SAVE_PATH = args.ckpt_save_path
BATCH_SIZE = args.batch_size
PRETRAINED_WEIGHTS = args.pretrained_weights
GRAY_SCALE = False if args.gray_scale == 'False' else True
HG_BLOCKS = args.hg_blocks
END_RELU = False if args.end_relu == 'False' else True
NUM_LANDMARKS = args.num_landmarks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(CKPT_SAVE_PATH)

dataloaders, dataset_sizes = get_dataset(VAL_IMG_DIR, VAL_LANDMARKS_DIR,
                                         BATCH_SIZE, NUM_LANDMARKS)
use_gpu = torch.cuda.is_available()
model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

if PRETRAINED_WEIGHTS != "None":
    checkpoint = torch.load(PRETRAINED_WEIGHTS)
    if 'state_dict' not in checkpoint:
        model_ft.load_state_dict(checkpoint)
    else:
        pretrained_weights = checkpoint['state_dict']
        model_weights = model_ft.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
        model_weights.update(pretrained_weights)
        model_ft.load_state_dict(model_weights)

model_ft = model_ft.to(device)

model_ft = eval_model(model_ft, dataloaders, dataset_sizes, writer, use_gpu, 1, 'val', CKPT_SAVE_PATH, NUM_LANDMARKS)

