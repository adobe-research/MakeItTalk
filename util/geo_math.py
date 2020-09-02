"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import numpy as np

def area_of_triangle(pts):

    AB = pts[1, :] - pts[0, :]
    AC = pts[2, :] - pts[0, :]

    return 0.5 * np.linalg.norm(np.cross(AB, AC))

def area_of_polygon(pts):
    l = pts.shape[0]
    s = 0
    for i in range(1, l-1):
        s += area_of_triangle(pts[(0, i, i+1), :])
    return s

def area_of_signed_triangle(pts):

    AB = pts[1, :] - pts[0, :]
    AC = pts[2, :] - pts[0, :]

    return 0.5 * np.cross(AB, AC)

def area_of_signed_polygon(pts):
    l = pts.shape[0]
    s = 0
    for i in range(1, l-1):
        s += area_of_signed_triangle(pts[(0, i, i+1), :])
    return s