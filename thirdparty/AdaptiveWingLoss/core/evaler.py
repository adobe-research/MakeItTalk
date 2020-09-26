import matplotlib
matplotlib.use('Agg')
import math
import torch
import copy
import time
from torch.autograd import Variable
import shutil
from skimage import io
import numpy as np
from utils.utils import fan_NME, show_landmarks, get_preds_fromhm
from PIL import Image, ImageDraw
import os
import sys
import cv2
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_model(model, dataloaders, dataset_sizes,
               writer, use_gpu=True, epoches=5, dataset='val',
               save_path='./', num_landmarks=68):
    global_nme = 0
    model.eval()
    for epoch in range(epoches):
        running_loss = 0
        step = 0
        total_nme = 0
        total_count = 0
        fail_count = 0
        nmes = []
        # running_corrects = 0

        # Iterate over data.
        with torch.no_grad():
            for data in dataloaders[dataset]:
                total_runtime = 0
                run_count = 0
                step_start = time.time()
                step += 1
                # get the inputs
                inputs = data['image'].type(torch.FloatTensor)
                labels_heatmap = data['heatmap'].type(torch.FloatTensor)
                labels_boundary = data['boundary'].type(torch.FloatTensor)
                landmarks = data['landmarks'].type(torch.FloatTensor)
                loss_weight_map = data['weight_map'].type(torch.FloatTensor)
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.to(device)
                    labels_heatmap = labels_heatmap.to(device)
                    labels_boundary = labels_boundary.to(device)
                    loss_weight_map = loss_weight_map.to(device)
                else:
                    inputs, labels_heatmap = Variable(inputs), Variable(labels_heatmap)
                    labels_boundary = Variable(labels_boundary)
                labels = torch.cat((labels_heatmap, labels_boundary), 1)
                single_start = time.time()
                outputs, boundary_channels = model(inputs)
                single_end = time.time()
                total_runtime += time.time() - single_start
                run_count += 1
                step_end = time.time()
                for i in range(inputs.shape[0]):
                    print(inputs.shape)
                    img = inputs[i]
                    img = img.cpu().numpy()
                    img = img.transpose((1, 2, 0)) #*255.0
                    # img = img.astype(np.uint8)
                    # img = Image.fromarray(img)
                    # pred_heatmap = outputs[-1][i].detach().cpu()[:-1, :, :]
                    pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
                    pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
                    pred_landmarks = pred_landmarks.squeeze().numpy()

                    gt_landmarks = data['landmarks'][i].numpy()
                    print(pred_landmarks, gt_landmarks)
                    import cv2
                    while(True):
                        imgshow = vis_landmark_on_img(cv2.UMat(img), pred_landmarks*4)
                        cv2.imshow('img', imgshow)

                        if(cv2.waitKey(10) == ord('q')):
                            break


                    if num_landmarks == 68:
                        left_eye = np.average(gt_landmarks[36:42], axis=0)
                        right_eye = np.average(gt_landmarks[42:48], axis=0)
                        norm_factor = np.linalg.norm(left_eye - right_eye)
                        # norm_factor = np.linalg.norm(gt_landmarks[36]- gt_landmarks[45])

                    elif num_landmarks == 98:
                        norm_factor = np.linalg.norm(gt_landmarks[60]- gt_landmarks[72])
                    elif num_landmarks == 19:
                        left, top = gt_landmarks[-2, :]
                        right, bottom = gt_landmarks[-1, :]
                        norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
                        gt_landmarks = gt_landmarks[:-2, :]
                    elif num_landmarks == 29:
                        # norm_factor = np.linalg.norm(gt_landmarks[8]- gt_landmarks[9])
                        norm_factor = np.linalg.norm(gt_landmarks[16]- gt_landmarks[17])
                    single_nme = (np.sum(np.linalg.norm(pred_landmarks*4 - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor

                    nmes.append(single_nme)
                    total_count += 1
                    if single_nme > 0.1:
                        fail_count += 1
                if step % 10 == 0:
                    print('Step {} Time: {:.6f} Input Mean: {:.6f} Output Mean: {:.6f}'.format(
                        step, step_end - step_start,
                        torch.mean(labels),
                        torch.mean(outputs[0])))
                # gt_landmarks = landmarks.numpy()
                # pred_heatmap = outputs[-1].to('cpu').numpy()
                gt_landmarks = landmarks
                batch_nme = fan_NME(outputs[-1][:, :-1, :, :].detach().cpu(), gt_landmarks, num_landmarks)
                # batch_nme = 0
                total_nme += batch_nme
        epoch_nme = total_nme / dataset_sizes['val']
        global_nme += epoch_nme
        nme_save_path = os.path.join(save_path, 'nme_log.npy')
        np.save(nme_save_path, np.array(nmes))
        print('NME: {:.6f} Failure Rate: {:.6f} Total Count: {:.6f} Fail Count: {:.6f}'.format(epoch_nme, fail_count/total_count, total_count, fail_count))
    print('Evaluation done! Average NME: {:.6f}'.format(global_nme/epoches))
    print('Everage runtime for a single batch: {:.6f}'.format(total_runtime/run_count))
    return model


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''

    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

    draw_curve(list(range(0, 32)))  # jaw
    draw_curve(list(range(33, 41)), color=(0, 0, 255), loop=True)  # eye brow
    draw_curve(list(range(42, 50)), color=(0, 0, 255), loop=True)
    draw_curve(list(range(51, 59)))  # nose
    draw_curve(list(range(60, 67)), loop=True)  # eyes
    draw_curve(list(range(68, 75)), loop=True)
    draw_curve(list(range(76, 87)), loop=True, color=(0, 255, 255))  # mouth
    draw_curve(list(range(88, 95)), loop=True, color=(255, 255, 0))

    return img