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


class ViewMorphing(nn.Module):
	def __init__(self, image_dim):
		x = np.arange(image_dim)
		y = np.arange(image_dim)
		q = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
		self.q = torch.LongTensor(q)

	def flatten(self, x):
    	N, C, H, W = x.size() # read in N, C, H, W
    	return x.view(N, C, -1)

	def forward(self, im1, im2, C, M1, M2):
		Cflat = flatten(C)
		embed = torch.nn.Embedding()
		for img in im1, im2:
			if img is im1:
				samp_2d = self.q + Cflat
			else:
				samp_2d = self.q - Cflat

			samp_flat = samp_2d[:, 0] + 224 * samp_2d[:, 1] 
			imflat = torch.transpose(flatten(img), 1, 2)
			





