import glob
import os
from scipy.misc import imread

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

NUM_TRAIN = 49000
NUM_VAL = 1000

DATA_DIR = "../src/preprocess/prep_res"

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

def load_dataset():
    ground_truths = []
    inputs = []
    for dir in os.listdir(DATA_DIR):
        src_p = DATA_DIR + "/" + dir
        if not os.path.isdir(src_p):
            continue
        src_f = sorted(glob.glob(src_p + "/*.png"))

        if not all(os.path.exists(f) for f in src_f): continue

        assert (len(src_f) % 3 == 0)

        for zero, truth, one in zip(*[iter(src_f)]*3):
            t = imread(truth)
            z = imread(zero)
            o = imread(one)
            ground_truths.append(t)
            inputs.append(np.concatenate((z,o), axis=2))

        return inputs, ground_truths


def main():
    inputs, gold = load_dataset()
    print (len(inputs))
    print (len(gold))

if __name__ == "__main__":
    main()
