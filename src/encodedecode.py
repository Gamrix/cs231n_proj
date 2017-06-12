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

"""
Example usage:
ed = EncodeDecode()
M1, M2, C = ed(im1, im2)
"""
class EncodeDecode(nn.Module):
    def __init__(self):
        super(EncodeDecode, self).__init__()

        self.ec3 = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=9, stride=1, padding=4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True))

        self.ec3feature = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, stride=1),
                nn.ReLU(inplace=True))

        self.ec4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True))

        self.ec4feature = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=1),
                nn.ReLU(inplace=True))

        self.ec5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True))

        self.ec5feature = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.ReLU(inplace=True))

        self.ec6 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 1024, kernel_size=1, stride=1),
                nn.ReLU(inplace=True))

        self.vc3sig = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid())

        self.cd1 = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 2048, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(2048, 768, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True))

        self.cd2 = nn.Sequential(
                nn.ConvTranspose2d(1024, 384, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True)
                )

        self.cd3 = nn.Sequential(
                nn.ConvTranspose2d(512, 192, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True)
                )

        self.cc3 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
                )


    def forward(self, im):
        #concat = torch.cat((im1, im2), dim=1)
        ec3 = self.ec3(im)
        print("ec3:", ec3.size())
        ec4 = self.ec4(ec3)
        print(ec4.size())
        ec5 = self.ec5(ec4)
        print(ec5.size())
        ec6 = self.ec6(ec5)

        M1 = self.vc3sig(ec6)
        # already a sigmoid

        cd1 = self.cd1(ec6)
        ec5feature = self.ec5feature(ec5)
        #print(cd1.size(), ec5.size(), ec5feature.size())
        cd1ec5feature = torch.cat((cd1, ec5feature), dim=1)
        cd2 = self.cd2(cd1ec5feature)
        ec4feature = self.ec4feature(ec4)
        cd2ec4feature = torch.cat((cd2, ec4feature), dim=1)
        cd3 = self.cd3(cd2ec4feature)
        ec3feature = self.ec3feature(ec3)
        cd3ec3feature = torch.cat((cd3, ec3feature), dim=1)
        C = self.cc3(cd3ec3feature)

        return im[:,:3], im[:,3:], C, M1, (1 - M1)
