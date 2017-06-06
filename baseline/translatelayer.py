

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
import torch.optim as optim
from torch.autograd import Variable

import itertools

class TranslateLayer(nn.Module):

    def __init__(self, in_channels, cell_size, stride=None, *args, **kwargs):
        if stride is None:
            stride = cell_size

        # make sure cell size is a power of 2
        # hate me all you want....
        cell_size_base = len(bin(cell_size)) - bin(cell_size).rfind('1')
        assert 2 ** cell_size_base == cell_size , "Cell size needs to be a power of 2"

        layers = [
            nn.Conv2d(in_channels, 9, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(9),
            # nn.Dropout2d(p=DROPOUT),
        ]

        for i in range(cell_size_base - 2):
            layers += [
                nn.Conv2d(9, 9, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(9),
            ]

        layers.append(nn.Conv2d(9, 9, kernel_size=5, stride=2, padding=2))
        self.conv_condense = nn.Sequential(*layers)

        super(TranslateLayer, self).__init__(*args, **kwargs)

