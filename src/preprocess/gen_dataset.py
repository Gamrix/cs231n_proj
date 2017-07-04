from __future__ import division, print_function

import logging
import os
import random
from multiprocessing import pool

from scipy.misc import imread, imsave, imresize

"""
Put folders inside of ./dataset/ to process
"""

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S', level=logging.INFO)

output_sz = (224, 224)
# input_folder = "./dataset"
output_folder = "./prep_res"
input_folder = "J:/kitti/dataset"
# output_folder = "J:/kitti/prep_res"


frame_interval = 3, 7
simple = False

if simple:
    output_folder = output_folder + "/simple"
    frame_interval = (1, 2)
else:
    output_folder = output_folder + "/normal"


def convert_folder_pics(folder_name):
    try:
        if not os.path.isdir(input_folder + "/" + folder_name): return

        if os.path.exists("{}/{}/finished.txt".format(output_folder, folder_name)):
            return

        logging.info("starting " + folder_name)
        src_p = input_folder + "/" + folder_name
        all_files = list(sorted(os.listdir(src_p)))
        # just hardcode the file names for now

        num_samples = len(all_files) // 2
        if simple: num_samples *= 2

        full_out_folder = output_folder + "/" + folder_name
        os.makedirs(full_out_folder, exist_ok=True)

        for i in range(num_samples):
            if i % 10 == 0:
                logging.info("Making image: {}".format(i))

            # choosing files to convert
            start_num = random.randrange(len(all_files))
            interval = random.randrange(*frame_interval)
            files = ["{:010d}.png".format(n + start_num) for n in (0, interval, 2 * interval)]
            src_f = [src_p + "/" + f for f in files]
            if not all(os.path.exists(f) for f in src_f): continue

            # choose an image offset
            w_offset = random.randint(0, 1392 - 512)

            for f, name in zip(files, ("src_0", "src_1_gt", "src_2")):
                orig_image = imread(src_p + "/" + f)

                im_height = orig_image.shape[0]
                crop = orig_image[:, w_offset:im_height + w_offset, :]
                resized = imresize(crop, output_sz)

                res_name = "{}/{}/{}__{}.jpg".format(output_folder, folder_name, i, name)
                imsave(res_name, resized)

        with open("{}/{}/finished.txt".format(output_folder, folder_name), "w") as f:
            f.write("Done")
    except Exception:
        import traceback

        logging.error("Failure with {}\n".format(folder_name) + traceback.format_exc())


if __name__ == '__main__':
    with pool.Pool(4) as p:
        p.map(convert_folder_pics, os.listdir(input_folder))
