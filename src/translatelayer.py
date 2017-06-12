from __future__ import print_function, division

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
        super(TranslateLayer, self).__init__(*args, **kwargs)
        if stride is None:
            stride = cell_2_pow

        # make sure cell size is a power of 2
        # hate me all you want....
        # cell_size_base = len(bin(cell_size)) - bin(cell_size).rfind('1')
        # assert 2 ** cell_size_base == cell_size , "Cell size needs to be a power of 2"
        # assert cell_2_pow >= 2, "Not built for 1 pixel tranforms yet."
        self.cell_2_pow = cell_2_pow
        self.in_channels = ctrl_in_channels

        self.conv_condense = nn.Sequential(
            nn.Conv2d(ctrl_in_channels, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.Conv2d(50, 18, kernel_size=3, padding=1)
        )

    def forward(self, image_pipe, ctrl_pipe):
        # will not concat img and ctrl pipes, another layer can do that
        assert(ctrl_pipe.size()[1] == self.in_channels)
        cell_sz = 2 ** self.cell_2_pow

        # build predictions
        conv_condense_out = self.conv_condense.forward(ctrl_pipe)

        # now build the translation mechanism

        B, _, img_H, img_W = image_pipe.size()
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

        # (bat x h x w x 2) x 3 x 9   orig image
        # (bat x h x w x 2) x 9 x 1   Filter
        # (bat x h x w x 2) x 3 x 1   res

        # clone there because .view needs a contiguous array.
        img_res_2 = img_res.view(B, 2, 3, img_rH, img_rW, 9).permute(0, 1, 3, 4, 2, 5).clone()
        img_flat = img_res_2.view(-1, *img_res_2.size()[-2:])

        b, c, n_h, n_w = conv_condense_out.size()
        translate = conv_condense_out.view(b, 2, 9, n_h, 1, n_w, 1).expand(b, 2, 9, n_h, cell_sz, n_w, cell_sz)
        translate = translate.permute(0, 1, 3, 4, 5, 6, 2).clone()
        translate = translate.view(b * 2 * n_h * cell_sz * n_w * cell_sz, 9, 1)

        # print(img_flat.size(), translate.size())
        raw_res = torch.bmm(img_flat, translate)

        # final result I want bat x 6 x h x w
        res_all = raw_res.view(b, 2, n_h* cell_sz, n_w * cell_sz, 3).permute(0, 1, 4, 2, 3).clone()
        res_all = res_all.view(b, 6, n_h* cell_sz, n_w * cell_sz)
        res = res_all[:, :, :img_H, :img_W]
        return res


class TrimLayer(nn.Module):
    def forward(self, img):
        return img[:, :, :-1, :-1]

class TranslateModel(nn.Module):
    def __init__(self):
        super(TranslateModel, self).__init__()

        # Next thing: Implement batch norm

        self.ec2 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU())

        # This results in 128 x 56 x 56  vector

        def conv_relu(start_dim, end_dim):
            return nn.Sequential(
                nn.Conv2d(start_dim, end_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(end_dim),
                nn.ReLU())

        def conv_squeeze(start_dim, end_dim):
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(start_dim, end_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(end_dim),
                nn.ReLU())

        self.ec3 = conv_squeeze(128, 256)  # 256 x 28 x 28  cellsz: 8
        self.ec4 = conv_squeeze(256, 512)  # 512 x 14 x 14  cellsz: 16
        self.ec5 = conv_squeeze(512, 512)  # 512 x 7 x 7   cellsz: 32
        self.ec6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())  # 1024 x 4 x 4


        # self.ec3feature = conv_relu(128, 64)
        # self.ec4feature = conv_relu(256, 128)
        # self.ec5feature = conv_relu(512, 256)
        # self.ec6feature = conv_relu(512, 256)


        self.refeature = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.refeature_final = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cd6_2 = nn.Sequential(
            nn.Conv2d(1024 + 128, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.cd3_2 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.cd5 = nn.Sequential(
                nn.Conv2d(1024 + 128, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, dilation=2),
                TrimLayer(),   # needed for the 7x7 matrix.
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            )

        def conv_transpose_conv(input_dim, output_dim):
            return nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=2, padding=1, dilation=2),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
            )

        self.cd4 = conv_transpose_conv(1024 + 128, 512)
        self.cd3 = conv_transpose_conv(512 + 128, 256)
        self.cd2 = conv_transpose_conv(256 + 128, 128)
        self.cd1 = conv_transpose_conv(128 + 64, 64)
        self.cd0 = conv_transpose_conv(64 + 64, 64)

        self.trans6 = TranslateLayer(1024, 6)
        self.trans6_2 = TranslateLayer(1024, 6)
        self.trans5 = TranslateLayer(1024, 5)
        self.trans4 = TranslateLayer(512, 4)
        self.trans3 = TranslateLayer(256, 3)
        self.trans3_2 = TranslateLayer(256, 3)
        self.trans2 = TranslateLayer(128, 2)
        self.trans1 = TranslateLayer(64, 1)
        self.trans0 = TranslateLayer(64, 0)

        # To create the mask
        self.mask = nn.Sequential(
            nn.Conv2d(64 + 6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )  # The other mask is just 1- mask



    def forward(self, im):

        # ec: Encode features (condensing of shape)
        # cd: Decode of features (uncondensing)

        #concat = torch.cat((im1, im2), dim=1)
        ec2 = self.ec2(im)
        ec3 = self.ec3(ec2)
        ec4 = self.ec4(ec3)
        ec5 = self.ec5(ec4)
        ec6 = self.ec6(ec5)

        cur_cd = ec6
        new_im = im
        groups = [(self.trans6, self.cd6_2, 6), (self.trans6_2, self.cd5, 6),
                  (self.trans5, self.cd4, 5), (self.trans4, self.cd3, 4),
                  (self.trans3, self.cd3_2, 3), (self.trans3_2, self.cd2, 3)]

        for img_trans, cd_trans, scale_f in groups:
            scale_factor = 2 ** (scale_f - 2)
            new_im = img_trans.forward(new_im, cur_cd)
            # Yes, this works for the ec6 case
            min_img = F.avg_pool2d(new_im, scale_factor, stride=scale_factor)
            im_feat = self.refeature(min_img)
            # print(im_feat.size(), cur_cd.size())
            next_cd = torch.cat((im_feat, cur_cd), dim=1)
            cur_cd = cd_trans.forward(next_cd)

        # now the final translation layers
        scale_factor = 4
        new_im = self.trans2.forward(new_im, cur_cd)
        min_img = F.avg_pool2d(new_im, scale_factor, stride=scale_factor)
        im_feat = self.refeature_final(min_img)
        next_cd = torch.cat((im_feat, cur_cd), dim=1)
        cur_cd = self.cd1.forward(next_cd)

        scale_factor = 2
        new_im = self.trans1.forward(new_im, cur_cd)
        min_img = F.avg_pool2d(new_im, scale_factor, stride=scale_factor)
        im_feat = self.refeature_final(min_img)
        next_cd = torch.cat((im_feat, cur_cd), dim=1)
        cur_cd = self.cd0.forward(next_cd)

        new_im = self.trans0.forward(new_im, cur_cd)
        # im_feat = self.refeature_final(new_im)
        next_cd = torch.cat((new_im, cur_cd), dim=1)

        M1 = self.mask.forward(next_cd)
        im0, im1 = new_im[:, :3], new_im[:, 3:]
        m1_e = M1.expand_as(im0)
        res = im0 * m1_e + im1 * (1 - m1_e)

        # match the interface of the encode_decode layer
        return res, 0, 0, im0, im1, M1, (1 - M1)


