"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
sys.path.append('thirdparty/AdaptiveWingLoss')
import numpy as np
import cv2
import torch
import torch.nn as nn
from thirdparty.AdaptiveWingLoss.core import models
from thirdparty.AdaptiveWingLoss.utils.utils import get_preds_fromhm
from dataset.image_translation.image_translation_dataset import vis_landmark_on_img98, vis_landmark_on_img
from models.model_image_translation import ResUnetGenerator
import  face_alignment

PRETRAINED_WEIGHTS = 'thirdparty/AdaptiveWingLoss/ckpt/WFLW_4HG.pth'
GRAY_SCALE = False
HG_BLOCKS = 4
END_RELU = False
NUM_LANDMARKS = 98

OLD_MODEL = False
LANDMARK_ALIGNMENT = 'FAN'

I2INET_DIR = 'ckpt_new_130'
IMAGE_NAME = 'donald.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if LANDMARK_ALIGNMENT == 'AWING':
    ''' Awings '''
    model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)
    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    if 'state_dict' not in checkpoint:
        model_ft.load_state_dict(checkpoint)
    else:
        pretrained_weights = checkpoint['state_dict']
        model_weights = model_ft.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
        model_weights.update(pretrained_weights)
        model_ft.load_state_dict(model_weights)
    print('Load AWing model sucessfully')
    fa_model = model_ft.to(device).eval()

# always load FAN
''' FAN '''
predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)


G = ResUnetGenerator(input_nc=6, output_nc=3, num_downs=6, use_dropout=False)
ckpt = torch.load(r'C:\Users\Yang Zhou\Downloads\{}.pth'.format(I2INET_DIR), map_location=device)
try:
    G.load_state_dict(ckpt['G'])
except:
    tmp = nn.DataParallel(G)
    tmp.load_state_dict(ckpt['G'])
    G.load_state_dict(tmp.module.state_dict())
    del tmp
print('Load G model sucessfully')
G = G.to(device).eval()


cap = cv2.VideoCapture(0)

ref_in = cv2.imread('examples/{}'.format(IMAGE_NAME))
# ref_in = cv2.resize(ref_in, (256, 256))
ref_shapes = predictor.get_landmarks(ref_in)[0]
ref_w = np.max(ref_shapes[:, 0]) - np.min(ref_shapes[:, 0])
ref_c = np.mean(ref_shapes, axis=0, keepdims=True)

ref_in = ref_in.transpose((2, 1, 0) if OLD_MODEL else (2, 0, 1)).astype(np.float32)/255.0
ref_in = torch.tensor(ref_in, requires_grad=False).to(device)

fls_pre_list = []

with torch.no_grad():
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        h, w, _ = frame.shape
        print(h, w)
        frame = frame[:, (w-h)//2:(w+h)//2]
        frame = cv2.flip(frame, 1)

        if LANDMARK_ALIGNMENT == 'AWING':
            ''' Awings '''
            # =====================================================================================================
            # # pre-localize face region
            # pred_landmarks = predictor.get_landmarks(frame)[0]
            # l, b = np.min(pred_landmarks, axis=0)
            # r, t = np.max(pred_landmarks, axis=0)
            # z = 1.8 / 2
            # new_half_w = int(max(r-l, t-b) * z)
            # c = [min(max(0+new_half_w, (l+r)//2), h-1-new_half_w), min(max(0+new_half_w, (b+t)//2), h-1-new_half_w)]
            # c = [int(item) for item in c]
            # print(c)
            # frame0 = cv2.resize(frame[c[0]-new_half_w:c[0]+new_half_w, c[1]-new_half_w:c[1]+new_half_w], (256, 256))
            # ===========================================================================================================

            frame0 = cv2.resize(frame, (256, 256))
            frame = frame0.copy().transpose((2, 0, 1)).astype(np.float32) / 255.0
            inputs = torch.tensor(frame, requires_grad=False).unsqueeze(0).to(device)
            outputs, boundary_channels = fa_model(inputs)
            pred_heatmap = outputs[-1][:, :-1, :, :].detach().cpu()
            pred_landmarks, _ = get_preds_fromhm(pred_heatmap)
            pred_landmarks = pred_landmarks[0].numpy() * 4
            # pred_landmarks[:, 0], pred_landmarks[:, 1] = pred_landmarks[:, 1] * 1 , pred_landmarks[:, 0] * 1
            frame0 = vis_landmark_on_img98(frame0, pred_landmarks).astype(np.uint8)  # 98x2

        else:
            ''' FAN '''
            pred_landmarks = predictor.get_landmarks(frame)[0]
            frame0 = vis_landmark_on_img(frame, pred_landmarks, linewidth=2)  # 68x2
            frame0 = cv2.resize(frame0, (256, 256))
            pred_landmarks[:, 0], pred_landmarks[:, 1] = pred_landmarks[:, 1] * 256. / h, pred_landmarks[:, 0] * 256. / h

        fls_pre_list.append(pred_landmarks)
        if(len(fls_pre_list) > 2):
            pred_landmarks = np.mean(fls_pre_list, axis=0)
            fls_pre_list.pop(0)
        print(len(fls_pre_list))

        img_fl = np.ones(shape=(256, 256, 3)) * 255.0
        c = np.mean(pred_landmarks, axis=0 , keepdims=True)
        pred_landmarks = ((pred_landmarks - c) / (np.max(pred_landmarks[:, 0]) - np.min(pred_landmarks[:, 0])) * ref_w + c).astype(np.int)

        if LANDMARK_ALIGNMENT == 'AWING':
            img_fl = vis_landmark_on_img98(img_fl, pred_landmarks).transpose((2, 0, 1)).astype(np.float32) / 255.0
        else:
            img_fl = vis_landmark_on_img(img_fl, pred_landmarks).transpose((2, 1, 0)).astype(np.float32) / 255.0
        img_fl = torch.tensor(img_fl, requires_grad=False).to(device)



        image_in = torch.cat([img_fl, ref_in], dim=0).unsqueeze(0)
        g_out = G(image_in)
        g_out = torch.tanh(g_out)[0]
        g_out = g_out.detach().cpu().numpy().transpose((2, 1, 0) if OLD_MODEL else (1, 2, 0))*255.0
        g_out = g_out.astype(np.uint8)

        img_fl_vis = img_fl.detach().cpu().numpy().transpose((2, 1, 0) if OLD_MODEL else (1, 2, 0))*255.0
        img_fl_vis = img_fl_vis.astype(np.uint8)

        frame = np.concatenate([frame0, img_fl_vis, g_out], axis=1)
        frame = cv2.resize(frame, (512*3, 512))
        # Display the resulting frame
        cv2.imshow('frame',cv2.UMat(frame))
        if cv2.waitKey(1)== ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()