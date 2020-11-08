"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import torch.nn as nn
import torch.nn.init as init
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class ShapeParts:
    def __init__(self, np_pts):
        self.data = np_pts

    def part(self, idx):
        return Point(self.data[idx, 0], self.data[idx, 1])


class Record():
    def __init__(self, type_list):
        self.data, self.count = {}, {}
        self.type_list = type_list
        self.max_min_data = None
        for t in type_list:
            self.data[t] = 0.0
            self.count[t] = 0.0

    def add(self, new_data, c=1.0):
        for t in self.type_list:
            self.data[t] += new_data
            self.count[t] += c

    def per(self, t):
        return self.data[t] / (self.count[t] + 1e-32)

    def clean(self, t):
        self.data[t], self.count[t] = 0.0, 0.0

    def is_better(self, t, greater):
        if(self.max_min_data == None):
            self.max_min_data = self.data[t]
            return True
        else:
            if(greater):
                if(self.data[t] > self.max_min_data):
                    self.max_min_data = self.data[t]
                    return True
            else:
                if (self.data[t] < self.max_min_data):
                    self.max_min_data = self.data[t]
                    return True
        return False

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def vis_landmark_on_img(img, shape, linewidth=2):
    '''
    Visualize landmark on images.
    '''
    if (type(shape) == ShapeParts):
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape.part(i).x, shape.part(i).y), (shape.part(i + 1).x, shape.part(i + 1).y),
                         color, lineWidth)
            if (loop):
                cv2.line(img, (shape.part(idx_list[0]).x, shape.part(idx_list[0]).y),
                         (shape.part(idx_list[-1] + 1).x, shape.part(idx_list[-1] + 1).y), color, lineWidth)

        draw_curve(list(range(0, 16)))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 0, 255))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 0, 255))
        draw_curve(list(range(27, 35)))  # nose
        draw_curve(list(range(36, 41)), loop=True)  # eyes
        draw_curve(list(range(42, 47)), loop=True)
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))

    else:
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
            if (loop):
                cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                         (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

        draw_curve(list(range(0, 16)))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 0, 255))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 0, 255))
        draw_curve(list(range(27, 35)))  # nose
        draw_curve(list(range(36, 41)), loop=True)  # eyes
        draw_curve(list(range(42, 47)), loop=True)
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))

    return img


def vis_landmark_on_plt(fl,  x_offset=0.0, show_now=True, c='r'):
    def draw_curve(shape, idx_list, loop=False, x_offset=0.0, c=None):
        for i in idx_list:
            plt.plot((shape[i, 0] + x_offset, shape[i + 1, 0] + x_offset), (-shape[i, 1], -shape[i + 1, 1]), c=c, lineWidth=1)
        if (loop):
            plt.plot((shape[idx_list[0], 0] + x_offset, shape[idx_list[-1] + 1, 0] + x_offset),
                     (-shape[idx_list[0], 1], -shape[idx_list[-1] + 1, 1]), c=c, lineWidth=1)

    draw_curve(fl, list(range(0, 16)), x_offset=x_offset, c=c)  # jaw
    draw_curve(fl, list(range(17, 21)), x_offset=x_offset, c=c)  # eye brow
    draw_curve(fl, list(range(22, 26)), x_offset=x_offset, c=c)
    draw_curve(fl, list(range(27, 35)), x_offset=x_offset, c=c)  # nose
    draw_curve(fl, list(range(36, 41)), loop=True, x_offset=x_offset, c=c)  # eyes
    draw_curve(fl, list(range(42, 47)), loop=True, x_offset=x_offset, c=c)
    draw_curve(fl, list(range(48, 59)), loop=True, x_offset=x_offset, c=c)  # mouth
    draw_curve(fl, list(range(60, 67)), loop=True, x_offset=x_offset, c=c)

    if(show_now):
        plt.show()


def try_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

import numpy
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def get_puppet_info(DEMO_CH, ROOT_DIR):
    import numpy as np
    B = 5000
    # for wilk example
    if (DEMO_CH == 'wilk_old'):
        bound = np.array([-B, -B, -B, 459, -B, B+918, 419, B+918, B+838, B+918, B+838, 459, B+838, -B, 419, -B]).reshape(1, -1)
        # bound = np.array([0, 0, 0, 459, 0, 918, 419, 918, 838, 918, 838, 459, 838, 0, 419, 0]).reshape(1, -1)
        scale, shift = -0.005276414887140783, np.array([-475.4316, -193.53225])
    elif (DEMO_CH == 'sketch'):
        bound = np.array([-10000, -10000, -10000, 221, -10000, 10443, 232, 10443, 10465, 10443, 10465, 221, 10465, -10000, 232, -10000]).reshape(1, -1)
        scale, shift = -0.006393177201290783, np.array([-226.8411, -176.5216])
    elif (DEMO_CH == 'onepunch'):
        bound = np.array([0, 0, 0, 168, 0, 337, 282, 337, 565, 337, 565, 168, 565, 0, 282, 0]).reshape(1, -1)
        scale, shift = -0.007558707536598317, np.array([-301.4903, -120.05265])
    elif (DEMO_CH == 'cat'):
        bound = np.array([0, 0, 0, 315, 0, 631, 299, 631, 599, 631, 599, 315, 599, 0, 299, 0]).reshape(1, -1)
        scale, shift = -0.009099476040795225, np.array([-297.17085, -259.2363])
    elif (DEMO_CH == 'paint'):
        bound = np.array([0, 0, 0, 249, 0, 499, 212, 499, 424, 499, 424, 249, 424, 0, 212, 0]).reshape(1, -1)
        scale, shift = -0.007409177996872789, np.array([-161.92345878, -249.40250103])
    elif (DEMO_CH == 'mulaney'):
        bound = np.array([0, 0, 0, 255, 0, 511, 341, 511, 682, 511, 682, 255, 682, 0, 341, 0]).reshape(1, -1)
        scale, shift = -0.010651548568731444, np.array([-333.54245, -189.081])
    elif (DEMO_CH == 'cartoonM_old'):
        bound = np.array([0, 0, 0, 299, 0, 599, 399, 599, 799, 599, 799, 299, 799, 0, 399, 0]).reshape(1, -1)
        scale, shift = -0.0055312373170456845, np.array([-398.6125, -240.45235])
    elif (DEMO_CH == 'beer'):
        bound = np.array([0, 0, 0, 309, 0, 618, 260, 618, 520, 618, 520, 309, 520, 0, 260, 0]).reshape(1, -1)
        scale, shift = -0.0054102709937112374, np.array([-254.1478, -156.6971])
    elif (DEMO_CH == 'color'):
        bound = np.array([0, 0, 0, 140, 0, 280, 249, 280, 499, 280, 499, 140, 499, 0, 249, 0]).reshape(1, -1)
        scale, shift = -0.012986159189209149, np.array([-237.27065, -79.2465])
    else:
        if (os.path.exists(os.path.join(ROOT_DIR, DEMO_CH + '.jpg'))):
            img = cv2.imread(os.path.join(ROOT_DIR, DEMO_CH + ".jpg"))
        elif (os.path.exists(os.path.join(ROOT_DIR, DEMO_CH + '.png'))):
            img = cv2.imread(os.path.join(ROOT_DIR, DEMO_CH + ".png"))
        else:
            print('not file founded.')
            exit(0)
        size = img.shape
        h = size[1] - 1
        w = size[0] - 1
        bound = np.array([-B, -B,
                          -B, w//4,
                          -B, w // 2,
                          -B, w//4*3,
                          -B, B + w,
                          h // 2, B+w,
                          B+h, B+w,
                          B+h, w // 2,
                          B+h, -B,
                          h//4, -B,
                          h // 2, -B,
                          h//4*3, -B]).reshape(1, -1)
        ss = np.loadtxt(os.path.join(ROOT_DIR, DEMO_CH + '_scale_shift.txt'))
        scale, shift = ss[0], np.array([ss[1], ss[2]])

    return bound, scale, shift


def close_input_face_mouth(shape_3d, p1=0.7, p2=0.5):
    shape_3d = shape_3d.reshape((1, 68, 3))
    index1 = list(range(60 - 1, 55 - 1, -1))
    index2 = list(range(68 - 1, 65 - 1, -1))
    mean_out = 0.5 * (shape_3d[:, 49:54] + shape_3d[:, index1])
    mean_in = 0.5 * (shape_3d[:, 61:64] + shape_3d[:, index2])
    shape_3d[:, 50:53] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, list(range(59 - 1, 56 - 1, -1))] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d[:, 49] -= (shape_3d[:, 61] - mean_in[:, 0]) * p2
    shape_3d[:, 53] -= (shape_3d[:, 63] - mean_in[:, -1]) * p2
    shape_3d[:, 59] -= (shape_3d[:, 67] - mean_in[:, 0]) * p2
    shape_3d[:, 55] -= (shape_3d[:, 65] - mean_in[:, -1]) * p2
    # shape_3d[:, 61:64] = shape_3d[:, index2] = mean_in
    shape_3d[:, 61:64] -= (shape_3d[:, 61:64] - mean_in) * p1
    shape_3d[:, index2] -= (shape_3d[:, index2] - mean_in) * p1
    shape_3d = shape_3d.reshape((68, 3))

    return shape_3d

def norm_input_face(shape_3d):
    scale = 1.6 / (shape_3d[0, 0] - shape_3d[16, 0])
    shift = - 0.5 * (shape_3d[0, 0:2] + shape_3d[16, 0:2])
    shape_3d[:, 0:2] = (shape_3d[:, 0:2] + shift) * scale
    face_std = np.loadtxt('src/dataset/utils/STD_FACE_LANDMARKS.txt').reshape(68, 3)
    shape_3d[:, -1] = face_std[:, -1] * 0.1
    shape_3d[:, 0:2] = -shape_3d[:, 0:2]

    return shape_3d, scale, shift

def add_naive_eye(fl):
    for t in range(fl.shape[0]):
        r = 0.95
        fl[t, 37], fl[t, 41] = r * fl[t, 37] + (1 - r) * fl[t, 41], (1 - r) * fl[t, 37] + r * fl[t, 41]
        fl[t, 38], fl[t, 40] = r * fl[t, 38] + (1 - r) * fl[t, 40], (1 - r) * fl[t, 38] + r * fl[t, 40]
        fl[t, 43], fl[t, 47] = r * fl[t, 43] + (1 - r) * fl[t, 47], (1 - r) * fl[t, 43] + r * fl[t, 47]
        fl[t, 44], fl[t, 46] = r * fl[t, 44] + (1 - r) * fl[t, 46], (1 - r) * fl[t, 44] + r * fl[t, 46]

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
                fl[t0, index] = r * fl[t1, index] + (1 - r) * fl[t2, index]

        for t0 in range(t - K1 + 1, t):
            interp_fl(t0, t - K1, t, r=(t - t0) / 1. / K1)
        for t0 in range(t + 1, t + K2):
            interp_fl(t0, t, t + K2, r=(t + K2 - 1 - t0) / 1. / K2)

    return fl