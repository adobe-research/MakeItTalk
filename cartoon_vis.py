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
import matplotlib.pyplot as plt

import shutil
from util.icp import icp
import os

DEBUG_MODE = False
ADD_NAIVE_EYE = False
GEN_AUDIO = True
GEN_FLS = True

trg_dir = 'examples_cartoon/sa_exp'
fls_names = glob.glob1(trg_dir, 'fake_fls_0_t001004_pos_ep_*.txt')
fls_names.sort()

for i in range(0,1):

    fl = np.loadtxt(os.path.join(trg_dir, fls_names[i])).reshape((-1, 68,3))


    DEMO_CH = 'std'
    output_dir = 'examples_cartoon/sa_exp'
    try:
        os.mkdir(output_dir)
    except:
        pass


    scale, shift = 0.01, np.array([-200, -250])
    fls = fl.reshape((-1, 68, 3))
    fls[:, :, 0:2] = -fls[:, :, 0:2]
    fls[:, :, 0:2] = (fls[:, :, 0:2] / scale)
    fls[:, :, 0:2] -= shift.reshape(1, 2)

    plt.plot(fls[0, :, 0], fls[0, :, 1], 'r-')
    plt.show()

    frame = np.ones((400, 400, 3), dtype=np.uint8) * 255
    for i in range(68):
        p = fls[0, i, 0:2]
        cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
    cv2.imwrite('vis.jpg', frame)

    r = list(range(0, 68))
    fls = fls[:, r, :]
    fls = fls[:, :, 0:2].reshape(-1, 68 * 2)
    # fls = fls.reshape(-1, 160)
    # r = list(range(0, 48)) + list(range(60, 68))

    np.savetxt(os.path.join(output_dir, 'warped_points.txt'), fls, fmt='%.2f')

    # static_points.txt
    static_frame = fls[0].reshape((-1, 2))
    np.savetxt(os.path.join(output_dir, 'reference_points.txt'), static_frame, fmt='%.2f')

    # triangle_vtx_index.txt
    shutil.copy(os.path.join('examples_cartoon', 'cartoonM' + '_delauney_tri.txt'),
                os.path.join(output_dir, 'triangulation.txt'))

    # ==============================================
    # Step 4 : Jukub's morphing
    # ==============================================
    warp_exe = os.path.join(os.getcwd(), 'facewarp', 'dingwarp.exe')
    import os

    if (os.path.exists(os.path.join(output_dir, 'output'))):
        shutil.rmtree(os.path.join(output_dir, 'output'))
    os.mkdir(os.path.join(output_dir, 'output'))
    os.chdir('{}'.format(os.path.join(output_dir, 'output')))
    print(os.getcwd())

    # os.system('{} {} {} {} {} {}'.format(
    #     warp_exe,
    #     os.path.join('vis.jpg'),
    #     os.path.join(output_dir, 'triangulation.txt'),
    #     os.path.join(output_dir, 'reference_points.txt'),
    #     os.path.join(output_dir, 'warped_points.txt'),
    #     # os.path.join(ROOT_DIR, 'puppets', sys.argv[6]),
    #     '-novsync -dump'))
    # os.system('ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" {}'.format(
    #     os.path.join(output_dir, 'sa_exp')
    # ))

    # shutil.copy(os.path.join(nn_result_dir, sys.argv[8]), os.path.join(nn_result_dir, '../../demo_result', sys.argv[8]))


    # MACOS
    # WINARCH=win64 WINEPREFIX=~/.wine-64prefix wine ../dingwarp.exe ../onepunch.png ../triangulation.txt ../reference_points.txt ../warped_points.txt ../onepunch_onlybody.jpg -novsync -dump
    # ffmpeg -y -r 62.5 -f image2 -i "%06d.tga" -i ../tim_02_16k.wav -shortest -pix_fmt yuv420p ../c02_7.mp4