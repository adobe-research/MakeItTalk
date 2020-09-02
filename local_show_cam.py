"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import numpy as np
import argparse
import platform

from models.model_image_translation import ResUnetGenerator, VGGLoss
import torch
import torch.nn as nn
import cv2
import os, glob
from dataset.image_translation.image_translation_dataset import vis_landmark_on_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if platform.release() == '4.4.0-83-generic':
    src_dir = r'/mnt/ntfs/Dataset/TalkingToon/VoxCeleb2_imagetranslation/raw_fl3d'
    mp4_dir = r'/mnt/ntfs/Dataset/VoxCeleb2/train_set/dev/mp4'
    jpg_dir = r'img_output'
    ckpt_dir = r'img_output'
    log_dir = r'img_output'
else: # 3.10.0-957.21.2.el7.x86_64
    # root = r'/mnt/nfs/scratch1/yangzhou/VoxCeleb2_imagetranslation'
    root = r'/mnt/nfs/scratch1/yangzhou/PreprocessedVox_imagetranslation'
    src_dir = os.path.join(root, 'raw_fl3d')
    # mp4_dir = r'/mnt/nfs/work1/kalo/yangzhou/VoxCeleb2/train_set/dev/mp4'
    mp4_dir = r'/mnt/nfs/scratch1/yangzhou/PreprocessedVox_mp4'
    jpg_dir = os.path.join(root, 'tmp_v')
    ckpt_dir = os.path.join(root, 'ckpt')
    log_dir = os.path.join(root, 'log')


''' Step 2. Train the network '''
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=1500, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_frames', type=int, default=1, help='')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')

parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--name', type=str, default='tmp')

parser.add_argument('--jpg_dir', type=str, default=jpg_dir)
parser.add_argument('--ckpt_dir', type=str, default=ckpt_dir)
parser.add_argument('--log_dir', type=str, default=log_dir)

parser.add_argument('--jpg_freq', type=int, default=60, help='')
parser.add_argument('--ckpt_last_freq', type=int, default=1800, help='')
parser.add_argument('--ckpt_epoch_freq', type=int, default=1, help='')

parser.add_argument('--load_G_name', type=str, default='')
parser.add_argument('--use_vox_dataset', type=str, default='raw')

parser.add_argument('--single_test', type=str, default='')

opt_parser = parser.parse_args()



class Image_translation_block():

    def __init__(self, opt_parser):
        print('Run on device {}'.format(device))

        for key in vars(opt_parser).keys():
            print(key, ':', vars(opt_parser)[key])
        self.opt_parser = opt_parser

        # model
        self.G = ResUnetGenerator(input_nc=6, output_nc=3, num_downs=6, use_dropout=False)

        ckpt = torch.load('/Users/yangzhou/Downloads/ckpt_145.pth', map_location=torch.device('cpu'))
        try:
            self.G.load_state_dict(ckpt['G'])
        except:
            tmp = nn.DataParallel(self.G)
            tmp.load_state_dict(ckpt['G'])
            self.G.load_state_dict(tmp.module.state_dict())
            del tmp

        self.G.to(device)


    def single_test(self):
        self.G.eval()

        jpg = cv2.imread('/Users/yangzhou/Downloads/xintong/tmp/taylor.jpg')
        fls = np.loadtxt('/Users/yangzhou/Downloads/xintong/tmp/fls.txt')
        fls = fls * 95
        fls[:, 0::3] += 130
        fls[:, 1::3] += 80

        writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mjpg'), 62.5, (256 * 3, 256))

        for i in range(fls.shape[0]//16):

            print(i, fls.shape[0]//16)

            img_fl = np.ones(shape=(256, 256, 3)) * 255
            fl = frame.astype(int)
            img_fl = vis_landmark_on_img(img_fl, np.reshape(fl, (68, 3)))
            frame = np.concatenate((img_fl, jpg), axis=2).astype(np.float32)/255.0

            image_in, image_out = np.swapaxes(frame, 0, 2), np.zeros(shape=(3, 256, 256))
            image_in, image_out = torch.tensor(image_in, requires_grad=False), \
                                  torch.tensor(image_out, requires_grad=False)

            image_in, image_out = image_in.reshape(-1, 6, 256, 256), image_out.reshape(-1, 3, 256, 256)
            image_in, image_out = image_in.to(device), image_out.to(device)

            g_out = self.G(image_in)
            g_out = torch.tanh(g_out)

            g_out = np.swapaxes(g_out.cpu().detach().numpy(), 1, 3)
            ref_in = np.swapaxes(image_in[:, 3:6, :, :].cpu().detach().numpy(), 1, 3)
            fls_in = np.swapaxes(image_in[:, 0:3, :, :].cpu().detach().numpy(), 1, 3)

            for i in range(g_out.shape[0]):
                frame = np.concatenate((ref_in[i], g_out[i], fls_in[i]), axis=1) * 255.0
                writer.write(frame.astype(np.uint8))

        writer.release()

        os.system('ffmpeg -y -i out.mp4 -pix_fmt yuv420p v.mp4')







model = Image_translation_block(opt_parser)
with torch.no_grad():
    model.single_test()
