

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

    def __init__(self, in_channels, cell_2_pow, stride=None, *args, **kwargs):
        if stride is None:
            stride = cell_2_pow

        # make sure cell size is a power of 2
        # hate me all you want....
        # cell_size_base = len(bin(cell_size)) - bin(cell_size).rfind('1')
        # assert 2 ** cell_size_base == cell_size , "Cell size needs to be a power of 2"
        assert cell_2_pow >= 2, "Not built for 1 pixel tranforms yet."
        self.cell_2_pow = cell_2_pow
        self.in_channels = in_channels

        layers = [
            nn.Conv2d(in_channels, 18, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(18),
            # nn.Dropout2d(p=DROPOUT),
        ]

        for i in range(cell_2_pow - 2):
            layers += [
                nn.Conv2d(18, 18, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(18),
            ]

        layers += [nn.Conv2d(18, 9, kernel_size=5, stride=2, padding=2),
                   nn.Softmax2d()]
        self.conv_condense = nn.Sequential(*layers)

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

        res = raw_res.view(b, n_h* cell_sz, n_w * cell_sz, 3).permute(0, 3, 1, 2)
        return res

        # final result I want bat x 3 x h x w



