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


class EncodeDecode(nn.Module):
    def __init__(self):
        self.ec3 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            nn.ReLU(inplace=True))

        self.ec4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True))

        self.ec5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True))

        self.ec6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True))

        self.vc3sig = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True))


    def forward(self, im1, im2):
        concat = torch.cat((im1, im2), dim=1)
        ec3 = self.ec3(concat)
        ec4 = self.ec4(ec3)
        ec5 = self.ec5(ec4)
        ec6 = self.ec6(ec5)

        M1 = self.vc3sig(ec6)

        return self.model(concat)


ed = EncodeDecode()
M1, M2, C = ed(im1, im2)
