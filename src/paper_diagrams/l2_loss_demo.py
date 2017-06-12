from __future__ import division, print_function

import numpy as np
import random
from scipy.misc import imread, imsave, imresize, imshow


photo_path = "0_9_gold.png"

full_img = imread(photo_path).astype(np.float64)



def calculate_losses(coords):
    size = 32


    def l2_loss(sample, true):
        return np.mean((true - sample) **2)

    def texture_loss(true, sample, offset=1):
        o = offset
        delta_x_samp = sample[:, o:] - sample[:, :-o]
        delta_x_true = true[:, o:] - true[:, :-o]
        delta_y_samp = sample[o:] - sample[:-o]
        delta_y_true = true[o:] - true[:-o]
        return l2_loss(delta_x_samp, delta_x_true) + l2_loss(delta_y_samp, delta_y_true)


    def crop(img, loc, size):
        return img[loc[1]: loc[1] + size, loc[0]:loc[0] + size]

    def magnify(img, factor):
        return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


    def print_metrics(orig, samp, description):
        imsave(description + ".png", magnify(flat_pic, 8))
        print()
        print(description)
        print("L2 Loss: ", l2_loss(orig, samp))

        texture_res = (texture_loss(orig, samp, i) for i in (1,2,4))
        print("Texture Loss: d=1: {}, d=2: {}, d=4: {}".format(*texture_res))

    orig_crop = crop(full_img, coords, size)
    imsave("orig.png", magnify(orig_crop, 8))

    average = np.mean(np.mean(orig_crop, axis=0, keepdims=True), axis=1, keepdims=True)
    flat_pic = magnify(average, 32)
    print_metrics(orig_crop, flat_pic, "average")


    shifted = crop(full_img, coords + [0, 2], size)
    print_metrics(orig_crop, shifted, "shifted_2px")

    shifted2 = crop(full_img, coords + [0, 4], size)
    print_metrics(orig_crop, shifted2, "shifted_4px")

    blend = (shifted2 + orig_crop)/2
    print_metrics(orig_crop, blend, "4px_blend")

    from itertools import product
    nearby = [crop(full_img, coords + [i, j], size) for i, j in product((-1, 0, 1), repeat=2)]
    blur = np.mean(nearby, axis=0)
    print_metrics(orig_crop, blur, "avg blur")



    # texture loss experiment


# coords = np.asarray([57, 155])
for i in range(5):
    coords = np.asarray([random.randrange(0, 224-32), random.randrange(0, 224-32)])
    print("______________________________________________________________________")
    print(coords)
    calculate_losses(coords)
