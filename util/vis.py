"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import ffmpeg

OTHER_SPECIFIC_VOICE = None

class Vis():

    def __init__(self, fls, filename, audio_filenam=None, fps=100, frames=1000):

        # from scipy.signal import savgol_filter
        # fls = savgol_filter(fls, 21, 3, axis=0)

        # adj nose
        # fls[:, 27 * 3:28 * 3] = fls[:, 28 * 3:29 * 3] * 2 - fls[:, 29 * 3:30 * 3]
        # fls[:, 28 * 3:29 * 3] = fls[:, 27 * 3:28 * 3]*0.75 + fls[:, 31 * 3:32 * 3]*0.25
        # fls[:, 29 * 3:30 * 3] = fls[:, 27 * 3:28 * 3]*0.5 + fls[:, 31 * 3:32 * 3]*0.5
        # fls[:, 30 * 3:31 * 3] = fls[:, 27 * 3:28 * 3] * 0.25 + fls[:, 31 * 3:32 * 3] * 0.75

        fls = fls * 120
        fls[:, 0::3] += 200
        fls[:, 1::3] += 100

        fls = fls.reshape((-1, 68, 3))
        fls = fls.astype(int)

        writer = cv2.VideoWriter(os.path.join('examples', 'tmp.mp4'),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (400, 400))

        frames = np.min((fls.shape[0], frames))
        for i in range(frames): #fls.shape[0]):
            # print(i, fls.shape[0])
            frame = np.ones((400, 400, 3), np.uint8) * 0
            frame = self.__vis_landmark_on_img__(frame, fls[i])
            writer.write(frame)
        writer.release()

        if(audio_filenam is not None):
            print(audio_filenam)
            os.system('ffmpeg -y -i {} -i {} -strict -2 -shortest {}'.format(
                os.path.join('examples', 'tmp.mp4'),
                audio_filenam,
                os.path.join('examples', '{}_av.mp4'.format(filename))
            ))
        else:
            os.system('ffmpeg -y -i {} {}'.format(
                os.path.join('examples', 'tmp.mp4'),
                os.path.join('examples', '{}_av.mp4'.format(filename))
            ))

        os.remove(os.path.join('examples', 'tmp.mp4'))




    def __vis_landmark_on_img__(self, img, shape, linewidth=2):
        '''
        Visualize landmark on images.
        '''
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
            if (loop):
                cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                         (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

        # draw_curve(list(range(0, 16)), color=(0, 255, 0))  # jaw
        # draw_curve(list(range(17, 21)), color=(0, 127, 255))  # eye brow
        # draw_curve(list(range(22, 26)), color=(0, 127, 255))
        # draw_curve(list(range(27, 35)), color=(255, 0, 0))  # nose
        # draw_curve(list(range(36, 41)), loop=True, color=(204, 0, 204))  # eyes
        # draw_curve(list(range(42, 47)), loop=True, color=(204, 0, 204))
        # draw_curve(list(range(48, 59)), loop=True, color=(0, 0, 255))  # mouth
        # draw_curve(list(range(60, 67)), loop=True, color=(0, 0, 255))
        # draw_curve(list(range(60, 64)), loop=False, color=(0, 0, 255))

        draw_curve(list(range(0, 16)), color=(0, 255, 0))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 255, 0))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 255, 0))
        draw_curve(list(range(27, 35)), color=(0, 255, 0))  # nose
        draw_curve(list(range(36, 41)), loop=True, color=(0, 255, 0))  # eyes
        draw_curve(list(range(42, 47)), loop=True, color=(0, 255, 0))
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))
        draw_curve(list(range(60, 64)), loop=False, color=(0, 0, 255))

        return img



class Vis_old():

    def __init__(self, run_name, pred_fl_filename, audio_filename, av_name='NAME', fps=100, frames=625,
                 postfix='', root_dir=r'E:\Dataset\TalkingToon\Obama', ifsmooth=True, rand_start=0):

        print(root_dir)
        self.src_dir = os.path.join(root_dir, r'nn_result/{}'.format(run_name))
        self.std_face = np.loadtxt(r'src/dataset/utils/STD_FACE_LANDMARKS.txt')
        self.std_face = self.std_face.reshape((-1, 204))

        fls = np.loadtxt(os.path.join(self.src_dir, pred_fl_filename))

        fls = fls * 120
        fls[:, 0::3] += 200
        fls[:, 1::3] += 100

        fls = fls.reshape((-1, 68, 3))
        fls = fls.astype(int)

        writer = cv2.VideoWriter(os.path.join(self.src_dir, 'tmp.mp4'),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (400, 400))

        frames = np.min((fls.shape[0], frames))
        for i in range(frames): #fls.shape[0]):
            # print(i, fls.shape[0])
            frame = np.ones((400, 400, 3), np.uint8) * 0
            frame = self.__vis_landmark_on_img__(frame, fls[i])
            writer.write(frame)
        writer.release()

        if(os.path.exists(os.path.join(root_dir, 'demo_wav', '{}'.format(audio_filename)))):
            ain = os.path.join(root_dir, 'demo_wav', '{}'.format(audio_filename))
        else:
            ain = os.path.join(root_dir, 'raw_wav', '{}'.format(audio_filename))
        # print(ain)
        # vin = ffmpeg.input(os.path.join(self.src_dir, 'tmp.mp4')).video
        # ain = ffmpeg.input(ain).audio
        # out = ffmpeg.output(vin, ain, os.path.join(self.src_dir, '{}_av.mp4'.format(pred_fl_filename[:-4])), shortest=None)
        # out = out.overwrite_output().global_args('-loglevel', 'quiet')
        # out.run()

        os.system('ffmpeg -y -loglevel error -i {} -ss {} {}'.format(
            ain, rand_start/62.5,
            os.path.join(self.src_dir, '{}_a_tmp.wav'.format(av_name))
        ))

        os.system('ffmpeg -y -loglevel error -i {} -i {} -pix_fmt yuv420p -strict -2 -shortest {}'.format(
            os.path.join(self.src_dir, 'tmp.mp4'),
            os.path.join(self.src_dir, '{}_a_tmp.wav'.format(av_name)),
            os.path.join(self.src_dir, '{}_av.mp4'.format(av_name))
        ))

        os.remove(os.path.join(self.src_dir, 'tmp.mp4'))
        os.remove(os.path.join(self.src_dir, '{}_a_tmp.wav'.format(av_name)))

        # os.remove(os.path.join(self.src_dir, filename))
        # exit(0)





    def __vis_landmark_on_img__(self, img, shape, linewidth=2):
        '''
        Visualize landmark on images.
        '''
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
            if (loop):
                cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                         (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

        # draw_curve(list(range(0, 16)), color=(0, 255, 0))  # jaw
        # draw_curve(list(range(17, 21)), color=(0, 127, 255))  # eye brow
        # draw_curve(list(range(22, 26)), color=(0, 127, 255))
        # draw_curve(list(range(27, 35)), color=(255, 0, 0))  # nose
        # draw_curve(list(range(36, 41)), loop=True, color=(204, 0, 204))  # eyes
        # draw_curve(list(range(42, 47)), loop=True, color=(204, 0, 204))
        # draw_curve(list(range(48, 59)), loop=True, color=(0, 0, 255))  # mouth
        # draw_curve(list(range(60, 67)), loop=True, color=(0, 0, 255))
        # draw_curve(list(range(60, 64)), loop=False, color=(0, 0, 255))

        draw_curve(list(range(0, 16)), color=(0, 255, 0))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 255, 0))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 255, 0))
        draw_curve(list(range(27, 35)), color=(0, 255, 0))  # nose
        draw_curve(list(range(36, 41)), loop=True, color=(0, 255, 0))  # eyes
        draw_curve(list(range(42, 47)), loop=True, color=(0, 255, 0))
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))
        draw_curve(list(range(60, 64)), loop=False, color=(0, 0, 255))

        return img


class Vis_comp():

    def __init__(self, run_name, pred1, pred2, audio_filename, av_name='NAME', fps=100, frames=625, postfix='', root_dir=r'E:\Dataset\TalkingToon\Obama', ifsmooth=True):

        print(root_dir)
        self.src_dir = os.path.join(root_dir, r'nn_result/{}'.format(run_name))
        self.std_face = np.loadtxt(r'src/dataset/utils/STD_FACE_LANDMARKS.txt')
        self.std_face = self.std_face.reshape((-1, 204))

        def fls_adj(fls):
            fls = fls * 120
            fls[:, 0::3] += 200
            fls[:, 1::3] += 100
            fls = fls.reshape((-1, 68, 3))
            fls = fls.astype(int)
            return fls

        fls = np.loadtxt(os.path.join(self.src_dir, pred1))
        fls2 = np.loadtxt(os.path.join(self.src_dir, pred2))
        fls = fls_adj(fls)
        fls2 = fls_adj(fls2)

        writer = cv2.VideoWriter(os.path.join(self.src_dir, 'tmp.mp4'),
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (400, 400))

        frames = np.min((fls.shape[0], frames))
        for i in range(frames): #fls.shape[0]):
            # print(i, fls.shape[0])
            frame = np.ones((400, 400, 3), np.uint8) * 0
            frame = self.__vis_landmark_on_img__(frame, fls[i])
            frame = self.__vis_landmark_on_img__(frame, fls2[i])
            writer.write(frame)
        writer.release()

        if(os.path.exists(os.path.join(root_dir, 'demo_wav', '{}'.format(audio_filename)))):
            ain = os.path.join(root_dir, 'demo_wav', '{}'.format(audio_filename))
        else:
            ain = os.path.join(root_dir, 'raw_wav', '{}'.format(audio_filename))

        os.system('ffmpeg -y -loglevel error -i {} -i {} -pix_fmt yuv420p -strict -2 -shortest {}'.format(
            os.path.join(self.src_dir, 'tmp.mp4'),
            ain,
            os.path.join(self.src_dir, '{}_av.mp4'.format(av_name))
        ))

        os.remove(os.path.join(self.src_dir, 'tmp.mp4'))


    def __vis_landmark_on_img__(self, img, shape, linewidth=2):
        '''
        Visualize landmark on images.
        '''
        def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
            for i in idx_list:
                cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, lineWidth)
            if (loop):
                cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                         (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, lineWidth)

        draw_curve(list(range(0, 16)), color=(0, 255, 0))  # jaw
        draw_curve(list(range(17, 21)), color=(0, 255, 0))  # eye brow
        draw_curve(list(range(22, 26)), color=(0, 255, 0))
        draw_curve(list(range(27, 35)), color=(0, 255, 0))  # nose
        draw_curve(list(range(36, 41)), loop=True, color=(0, 255, 0))  # eyes
        draw_curve(list(range(42, 47)), loop=True, color=(0, 255, 0))
        draw_curve(list(range(48, 59)), loop=True, color=(0, 255, 255))  # mouth
        draw_curve(list(range(60, 67)), loop=True, color=(255, 255, 0))
        draw_curve(list(range(60, 64)), loop=False, color=(0, 0, 255))

        return img