from __future__ import division, print_function

import os
import random

import numpy as np
from scipy.misc import imread, imsave, imresize


"""
Put folders inside of ./dataset/ to process
"""


output_sz = (224, 224)
input_folder = "./dataset"
output_folder = "./prep_res"

def convert_folder_pics(folder_name):
    src_p = input_folder + "/" + folder_name
    files = list(sorted(os.listdir(src_p)))
    # just hardcode the file names for now

    num_samples = len(files) // 2
    for i in range(num_samples):
        if 1 %10 == 0:
            print("Making image:", i)

        start_num = random.randrange(len(files))
        interval = random.randrange(3, 7)
        files = ["{:010d}.png".format(n + start_num) for n in (0, interval, 2 * interval)]
        src_f = [src_p + "/" + f for f in files]

        if not all(os.path.exists(f) for f in src_f): continue

        w_offset = random.randint(0, 1392 - 512)

        for f, name in zip(files, ("src_0", "ground_t", "src_1")):
            orig_image = imread(src_p + "/" + f)

            im_height = orig_image.shape[0]
            crop = orig_image[:, w_offset:im_height + w_offset, :]
            resized = imresize(crop, output_sz)
            res_name = output_folder + "/{}__{}.png".format(i, name)
            imsave(res_name, resized)


if __name__ == '__main__':
    for dir in os.listdir(input_folder):
        if os.path.isdir(input_folder + "/" + dir):
            convert_folder_pics(dir)

