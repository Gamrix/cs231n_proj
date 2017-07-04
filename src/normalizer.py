from __future__ import division, print_function

import numpy as np

norm_factor = 255 / 2


def normalize(image):
    return image.float() / norm_factor - 1


def denorm(norm_img):
    normed_data = (norm_img + 1) * norm_factor
    return np.clip(normed_data, 0, 255).astype("uint8")
