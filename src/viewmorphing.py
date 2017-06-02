import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import os

import timeit

dtype=torch.cuda.FloatTensor
dtypelong=torch.cuda.LongTensor

if os.path.exists("../john_local_flag.txt"):
    dtype=torch.FloatTensor
    dtypelong=torch.LongTensor

class ViewMorphing(nn.Module):
    def __init__(self, img_dim=224):
        super(ViewMorphing, self).__init__()
        self.image_dim = img_dim
        x = np.arange(self.image_dim)
        y = np.arange(self.image_dim)
        q = np.array([np.repeat(x, len(y)), np.tile(y, len(x))])
        self.q = Variable(torch.from_numpy(q).type(dtype))

    def flatten(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.contiguous().view(N, C, -1)

    def coordToInd(self, x):
        return (x[:, 1] + self.image_dim * x[:, 0]).type(dtypelong).detach()

    def get_pixel(self, point, neighbor, image):

        # weighting result pixel bilinearly
        weight = 1 - torch.abs(point - neighbor)
        weight = weight[:, 0] * weight[:, 1]

        inds = self.coordToInd(neighbor)
        a = torch.gather(image[:,0], 1, inds)
        b = torch.gather(image[:,1], 1, inds)
        c = torch.gather(image[:,2], 1, inds)
        pixel = torch.stack((a, b, c), dim=1)
        return weight.unsqueeze(1).expand_as(pixel) * pixel

    def get_masked_RP(self, image, mask, qi_orig):
        imflat = self.flatten(image)
        qi = torch.clamp(qi_orig, 0.001, self.image_dim - 1.001)
        res_img_flat = \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].floor(), qi[:,1:2].floor()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].ceil(), qi[:,1:2].floor()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].floor(), qi[:,1:2].ceil()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].ceil(), qi[:,1:2].ceil()), dim=1), imflat)

        # encourage some good gradients by penalizing for going out of bound
        res_img_flat = res_img_flat #/ (1 + torch.sum((qi_orig - qi) ** 2, dim=1)).unsqueeze(1).expand_as(res_img_flat)

        res_img = res_img_flat.view_as(image)
        return res_img * mask.expand_as(res_img)

    def forward(self, arglist):
        im1, im2, C, M1, M2 = arglist
        Cflat = self.flatten(C)

        a = self.get_masked_RP(im1, M1, self.q.expand_as(Cflat) + Cflat)
        b = self.get_masked_RP(im2, M2, self.q.expand_as(Cflat) - Cflat)

        return a + b


            
