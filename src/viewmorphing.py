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

dtype=torch.cuda.LongTensor

class ViewMorphing(nn.Module):
    def __init__(self):
        super(ViewMorphing, self).__init__()
        self.image_dim = 224
        x = np.arange(self.image_dim)
        y = np.arange(self.image_dim)
        q = np.array([np.repeat(x, len(y)), np.tile(y, len(x))])
        self.q = Variable(torch.from_numpy(q).type(dtype))

    def flatten(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.contiguous().view(N, C, -1)

    def coordToInd(self, x):
        return x[:, 0] + self.image_dim * x[:, 1]

    def get_pixel(self, point, neighbor, image):
        weight = torch.cumprod(1 - torch.abs(point - neighbor), dim=2)
        pixel = torch.index_select(image, 2, coordToInd(neighbor))
        return weight.extend_as(pixel) * pixel

    def get_masked_RP(self, image, mask, qi):
        imflat = self.flatten(img)
        res_img_flat = \
            self.get_pixel(qi, torch.cat((samp2d[:, 0].floor(), samp2d[:,1].floor()), dim=1), imflat) + \
            self.get_pixel(qi, torch.cat((samp2d[:, 0].ceil(), samp2d[:,1].floor()), dim=1), imflat) + \
            self.get_pixel(qi, torch.cat((samp2d[:, 0].floor(), samp2d[:,1].ceil()), dim=1), imflat) + \
            self.get_pixel(qi, torch.cat((samp2d[:, 0].ceil(), samp2d[:,1].ceil()), dim=1), imflat)

        res_img = res_img_flat.view(im1.size())
        return res_img * mask

    def forward(self, arglist):
        im1, im2, C, M1, M2 = arglist
        Cflat = self.flatten(C)

        return self.get_masked_RP(im1, M1, self.q.expand_as(Cflat) + Cflat) + \
            self.get_masked_RP(im2, M2, self.q.expand_as(Cflat) - Cflat)


            
