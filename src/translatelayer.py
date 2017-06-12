

# Background:

# Traditional CNN layers move information very slowly across the cnn
# For video interpolation, we need data to travel 30-100 pixels
# This kind of gradient without substantial information loss is
# not possible with traditional CNN architectures
# (or RNNs/LSTMs for that matter)

# Therefore, we need some way to transport data quickly across
# an image in a trainable fashion

# Transport layers is one idea that I have. Another is a varaint of global
# attention in RNN-based models.


# Idea of the Translate Layer
# Translate layers are an intermidiate between global attention, and local
# cnn layers.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import itertools

class TranslateLayer(nn.Module):

    def __init__(self, ctrl_in_channels, cell_2_pow, stride=None, *args, **kwargs):
        if stride is None:
            stride = cell_2_pow

        # make sure cell size is a power of 2
        # hate me all you want....
        # cell_size_base = len(bin(cell_size)) - bin(cell_size).rfind('1')
        # assert 2 ** cell_size_base == cell_size , "Cell size needs to be a power of 2"
        # assert cell_2_pow >= 2, "Not built for 1 pixel tranforms yet."
        self.cell_2_pow = cell_2_pow
        self.in_channels = ctrl_in_channels

        self.conv_condense = nn.Seqential(
            nn.Conv2d(ctrl_in_channels, 50, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(50),
            nn.Conv2d(50, 9, kernel_size=3, padding=1)
        )
        super(TranslateLayer, self).__init__(*args, **kwargs)

    def forward(self, image_pipe, ctrl_pipe):
        # will not concat img and ctrl pipes, another layer can do that
        assert(ctrl_pipe.size()[1] == self.in_channels)
        cell_sz = 2 ** self.cell_2_pow

        # build predictions
        conv_condense_out = self.conv_condense.forward(ctrl_pipe)

        # now build the translation mechanism

        img_H, img_W = image_pipe.size()[2:]
        padding_H = 0
        if img_H % cell_sz != 0:
            padding_H = cell_sz - img_H % cell_sz
        img_rH = padding_H + img_H

        padding_W = 0
        if img_W % cell_sz != 0:
            padding_W = cell_sz - img_W % cell_sz
        img_rW = padding_W + img_W

        # pad the image on all sides
        img = F.pad(image_pipe, (cell_sz, padding_W + cell_sz, cell_sz, padding_H + cell_sz))

        img_stack = []
        for i in range(3):
            w_start = i * cell_sz
            for j in range(3):
                h_start = j * cell_sz
                img_stack.append(img[:, :, h_start:h_start + img_rH, w_start:w_start + img_rW])

        img_res = torch.stack(img_stack, 4)

        # b x n x m
        # b x m x p

        # (bat x h x w) x 3 x 9   orig image
        # (bat x h x w) x 9 x 1   Filter
        # (bat x h x w) x 3 x 1   res
        img_res_2 = img_res.permute(0, 2, 3, 1, 4)
        img_flat = img_res_2.view(-1, *img_res_2.size()[-2:])

        b, c, n_h, n_w = conv_condense_out.size()
        translate = conv_condense_out.view(b, c, n_h, 1, n_w, 1).expand(b, c, n_h, cell_sz, n_w, cell_sz)
        translate = translate.permute(0, 2, 3, 4, 5, 1).view(b * n_h * cell_sz, n_w * cell_sz, c, 1)

        raw_res = torch.bmm(img_flat, translate)

        # final result I want bat x 3 x h x w
        res_all = raw_res.view(b, n_h* cell_sz, n_w * cell_sz, 3).permute(0, 3, 1, 2)
        res = res_all[:, :, :img_H, :img_W]
        return res


class TranslateModel(nn.Module):
    def __init__(self):
        super(TranslateModel, self).__init__()

        # Next thing: Implement batch norm

        self.ec3 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # condense
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # condense
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        # This results in 128 x 56 x 56  vector

        def conv_relu(start_dim, end_dim):
            return nn.Sequential(
                nn.Conv2d(start_dim, end_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(end_dim),
                nn.ReLU(inplace=True))

        def conv_squeeze(start_dim, end_dim):
            return nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(start_dim, end_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.ec4 = conv_squeeze(128, 256)  # 256 x 28 x 28  cellsz: 8
        self.ec5 = conv_squeeze(256, 512)  # 512 x 14 x 14  cellsz: 16
        self.ec6 = conv_squeeze(512, 512)  # 512 x 7 x 7   cellsz: 32
        self.ec7 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True))  # 1024 x 4 x 4


        # self.ec3feature = conv_relu(128, 64)
        # self.ec4feature = conv_relu(256, 128)
        # self.ec5feature = conv_relu(512, 256)
        # self.ec6feature = conv_relu(512, 256)

        self.refeature = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)

        self.cd7_2 = nn.Sequential(
            nn.Conv2d(1024 + 128, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cd6 = nn.Sequential(
            nn.Conv2d(1024 + 128, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, dilation=2),
            nn.ReLU(inplace=True),
        )  # result from this needs some trimming

        # To create the mask
        self.cd5 = nn.Sequential(
            nn.Conv2d(1024 + 128, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 768, kernel_size=4, stride=2, padding=1, dilation=2),
            nn.ReLU(inplace=True))

        self.cd4 = nn.Sequential(
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

        # ec: Encode features (condensing of shape)
        # cd: Decode of features (uncondensing)
        im0, im1 = im[:, :3], im[:, 3:]

        #concat = torch.cat((im1, im2), dim=1)
        ec3 = self.ec3(im)
        ec4 = self.ec4(ec3)
        ec5 = self.ec5(ec4)
        ec6 = self.ec6(ec5)
        ec7 = self.ec7(ec6)  # cellsz: 64


        im0 = ec7

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


