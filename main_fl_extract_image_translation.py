"""
 # Copyright 2020 Adobe
 # All Rights Reserved.
 
 # NOTICE: Adobe permits you to use, modify, and distribute this file in
 # accordance with the terms of the Adobe license agreement accompanying
 # it.
 
"""

import sys
import os, glob
import numpy as np
import cv2
import argparse
import platform

# # full vox celeb dataset
from dataset.image_translation.data_preparation import landmark_extraction
landmark_extraction(int(sys.argv[1]), int(sys.argv[2]))

# Preprocessed vox celeb dataset (refer to monkey-net
# from dataset.image_translation.data_preparation_with_preprocessing import landmark_extraction
# landmark_extraction(int(sys.argv[1]), int(sys.argv[2]))