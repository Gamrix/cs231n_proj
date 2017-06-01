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

    def forward(self, arglist):
        im1, im2, C, M1, M2 = arglist
        Cflat = self.flatten(C)
        Cflat = Cflat.type(dtype)
        #embed = torch.nn.Embedding(self.image_dim ** 2, 3)
        imgs = []
        for img, mask in (im1, M1), (im2, M2):
            if img is im1:
                print(self.q.size(), Cflat.size())
                samp_2d = self.q.expand_as(Cflat) + Cflat
            else:
                samp_2d = self.q.expand_as(Cflat) - Cflat

            samp_flat = samp_2d[:, 0] + self.image_dim * samp_2d[:, 1]
            imflat = torch.transpose(self.flatten(img), 1, 2)
            #embed.weight = imflat
            ##res_img_flat = embed(samp_flat)
            res_img_flat = torch.index_select(imflat, 0, samp_flat)
            res_img = res_img_flat.view(im1.size())
            # now we want to mask it

            imgs.append(res_img * mask)

        return imgs[0] + imgs[1]
