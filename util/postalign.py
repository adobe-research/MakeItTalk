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
import scipy.ndimage

fs = ['suit1_pred_fls_t7_audio_embed.mp4' ]

for f in fs:

    os.system('ffmpeg -y -i examples/{} -filter:v crop=256:256:256:0 -strict -2 examples/crop_{}'.format(f, f))

    cap = cv2.VideoCapture('examples/crop_{}'.format(f))
    writer = cv2.VideoWriter('examples/tmp_{}.mp4'.format(f[:-4]),
                                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 62.5, (256, 256))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    fir = np.copy(prvs)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.9,
                           minDistance = 3,
                           blockSize = 3)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_gray)
    mask[-50:, 128:] = 1

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
    p0 = p0[0:1]

    ori_ab = None

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    ii = 0
    while(ii>-1):
        print(f, ii, length)
        ii += 1

        ret,frame = cap.read()
        if(not ret):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            # frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            if(ori_ab is None):
                ori_ab = [a, b]

        # add dot
        # img = cv2.add(frame,mask)

        # rgb = img
        rgb = scipy.ndimage.shift(frame, shift=[ori_ab[1]-b, ori_ab[0]-a, 0], mode='reflect')

        # cv2.imshow('frame',rgb)
        writer.write(rgb)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()
    writer.release()

    f = f[:-4]
    os.system('ffmpeg -loglevel error -y -i {} -vn {}'.format(
        os.path.join('../examples', '{}.mp4'.format(f)), os.path.join('../examples', 'a_' + f + '.wav')
    ))

    os.system('ffmpeg -loglevel error -y -i {} -i {} -pix_fmt yuv420p -shortest -strict -2 {}'.format(
        os.path.join('../examples', 'tmp_{}.mp4'.format(f)), os.path.join('../examples', 'a_' + f + '.wav'),
        os.path.join('../examples', 'f_' + f + '.mp4')
    ))
    os.remove(os.path.join('../examples', 'tmp_{}.mp4'.format(f)))
    os.remove(os.path.join('../examples', 'a_' + f + '.wav'))
