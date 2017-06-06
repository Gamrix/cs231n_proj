from __future__ import division, print_function

import numpy as np
import os
import random

import numpy as np
from scipy.misc import imread, imsave, imresize
from multiprocessing import pool, Semaphore
from contextlib import contextmanager
import logging

# just converting a folder of pngs to jpegs

src_p  = "J:/kitti/dataset/2011_09_30_drive_0028_sync"
output_folder = "J:/kitti/converted_jpgs"

os.makedirs(output_folder, exist_ok=True)

for f in os.listdir(src_p):
    orig_image = imread(src_p + "/" + f)
    res_name = "{}/{}.jpg".format(output_folder, os.path.splitext(f)[0])
    imsave(res_name, orig_image)
