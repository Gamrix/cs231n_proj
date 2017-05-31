import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit


class ViewMorphing(nn.Module):
    def __init__(self, image_dim):
        x = np.arange(image_dim)
        y = np.arange(image_dim)
        q = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
        self.q = torch.LongTensor(q)

    def flatten(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, C, -1)

    def forward(self, im1, im2, C, M1, M2):
        N, C, H, W = im1.size()  # read in N, C, H, W
        img_dim = 224
        Cflat = self.flatten(C)
        embed = torch.nn.Embedding(img_dim ** 2, 3)
        imgs = []
        for img, mask in (im1, M1), (im2, M2):
            if img is im1:
                samp_2d = self.q + Cflat
            else:
                samp_2d = self.q - Cflat

            samp_flat = samp_2d[:, 0] + img_dim * samp_2d[:, 1]
            imflat = torch.transpose(self.flatten(img), 1, 2)
            embed.weight = imflat
            res_img_flat = embed(samp_flat)
            res_img = res_img_flat.view(N, C, H, W)
            # now we want to mask it

            imgs.append(res_img * mask)

        return imgs[0] + imgs[1]
