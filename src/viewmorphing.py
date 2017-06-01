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

dtype=torch.cuda.FloatTensor
dtypelong=torch.cuda.LongTensor

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
        return (x[:, 0] + self.image_dim * x[:, 1]).type(dtypelong).detach()

    def get_pixel(self, point, neighbor, image):
        print("init:", image.size(), neighbor.size())
        weight = 1 - torch.abs(point - neighbor)
        weight = weight[:, 0] * weight[:, 1]
        inds = self.coordToInd(neighbor)
        print("inds:", inds.size())
        a = torch.gather(image[:,0], 0, inds)
        b = torch.gather(image[:,1], 0, inds)
        c = torch.gather(image[:,2], 0, inds)
        print("abc:", a.size(), b.size(), c.size())
        pixel = torch.stack((a, b, c), dim=2)
        print("pixel:", pixel.size())
        #print(weight.size(), pixel.size())
        return weight * pixel

    def get_masked_RP(self, image, mask, qi):
        print("qi:", qi.size())
        imflat = self.flatten(image)
        res_img_flat = \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].floor(), qi[:,1:2].floor()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].ceil(), qi[:,1:2].floor()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].floor(), qi[:,1:2].ceil()), dim=1), imflat) + \
                self.get_pixel(qi, torch.cat((qi[:, 0:1].ceil(), qi[:,1:2].ceil()), dim=1), imflat)

        res_img = res_img_flat.view(im1.size())
        return res_img * mask

    def forward(self, arglist):
        im1, im2, C, M1, M2 = arglist
        Cflat = self.flatten(C)

        a = self.get_masked_RP(im1, M1, self.q.expand_as(Cflat) + Cflat)
        b = self.get_masked_RP(im2, M2, self.q.expand_as(Cflat) - Cflat)

        return a + b


            
