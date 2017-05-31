from __future__ import print_function, division

import glob
import os
from scipy.misc import imread, imsave

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from normalizer import normalize, denorm

NUM_TRAIN = 32000
NUM_VAL = 1600
NUM_SAVED_SAMPLES = 16
BATCH_SIZE = 64
DATA_DIR = "../src/preprocess/prep_res"
PRINT_EVERY = 1

NUM_EPOCHS = 3
DROPOUT = 0.15
INIT_LR = 5e-4

dtype=torch.cuda.FloatTensor

if os.path.exists("../john_local_flag.txt"):
    # this is because my local machine can't handle the batch size...
    BATCH_SIZE = 4
    NUM_EPOCHS = 1

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
    length = len(os.listdir(DATA_DIR))
    if os.path.isfile('saved_in_data') and os.path.isfile('saved_ground_truths'):
        print ("Reading cached numpy data in from file...")
        inputs = np.fromfile('saved_in_data')
        ground_truths = np.fromfile('saved_ground_truths')
        return inputs, ground_truths
    for i, dir in enumerate(os.listdir(DATA_DIR)):
        print ("\tOn dir %d of %d" % (i, length))
        src_p = DATA_DIR + "/" + dir
        if not os.path.isdir(src_p):
            continue
        src_f = sorted(glob.glob(src_p + "/*.jpg"))

        if not all(os.path.exists(f) for f in src_f): continue

        assert (len(src_f) % 3 == 0)

        for zero, truth, one in zip(*[iter(src_f)]*3):
            t = imread(truth)
            z = imread(zero)
            o = imread(one)
            ground_truths.append(t)
            inputs.append(np.concatenate((z,o), axis=2))

    inputs, ground_truths = np.array(inputs), np.array(ground_truths)

    print ("Caching numpy data for next run...")
    with open('saved_in_data', 'w+') as f:
        inputs.tofile(f)
    with open('saved_ground_truths', 'w+') as f:
        ground_truths.tofile(f)

    return inputs, ground_truths

def make_loaders(inputs, gold):
    inputs_t = torch.from_numpy(inputs).byte()
    gold_t = torch.from_numpy(gold).byte()
    dataset = TensorDataset(inputs_t, gold_t)

    train = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=ChunkSampler(NUM_TRAIN, 0))
    val = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
    test = None # For now

    return train, val, test

def train(model, loss_fn, optimizer, train_data, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d...' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(train_data):
            x_var = Variable(normalize(x).permute(0,3,1,2)).type(dtype)
            y_var = Variable(normalize(y).permute(0,3,1,2)).type(dtype)

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % PRINT_EVERY == 0:
                print('\tt = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def eval(model, dev_data, loss_fn):
    print("Running evaluation...")
    total_loss = 0.0
    model.eval()
    for t, (x, y) in enumerate(dev_data):
        x_var = Variable(normalize(x).permute(0,3,1,2)).type(dtype)
        y_var = Variable(normalize(y).permute(0,3,1,2)).type(dtype)
        
        scores = model(x_var)
        print(scores.size())
        for i in range(NUM_SAVED_SAMPLES):
            name = "./eval/{}_{}_".format(t, i)
            imsave(name + "gen.png", np.transpose(denorm(scores[i].data.cpu().numpy()), axes=[1,2,0]))
            imsave(name + "gold.png", np.transpose(denorm(y_var[i].data.cpu().numpy()), axes=[1,2,0]))
            x = x_var[i].data.cpu().numpy()
            imsave(name + "orig_0.png", x[:3,:,:])
            imsave(name + "orig_1.png", x[3:,:,:])
        
        total_loss += loss_fn(scores, y_var).data[0]

    print("Total eval loss: %.4f, Avg eval loss: %.4f" % (total_loss, total_loss / NUM_VAL))


def run_model(train_data, val_data, test_data):
    model_base = nn.Sequential (
        # Conv 1
        nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=(1,1), bias=True), # out 32 * 224 * 224
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=DROPOUT),
        # Conv 2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(1,1), bias=True), # out 64 * 224 * 224
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=DROPOUT),
        # Conv 3
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=(1,1), bias=True), # out 32 * 224 * 224
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=DROPOUT),
        # Conv 4
        nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=(1,1), bias=True), # out 16 * 224 * 224
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=DROPOUT),
        # Conv 5
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=(1,1), bias=True), # out 3 * 224 * 224
    ).type(dtype)
    
    loss_fn = nn.L1Loss()  # TODO: L2 loss
    optimizer = optim.Adam(model_base.parameters(), lr=INIT_LR)

    train(model_base, loss_fn, optimizer, train_data, num_epochs=NUM_EPOCHS) 
    eval(model_base, val_data, loss_fn)

def main():
    print ("Loading dataset...")
    inputs, gold = load_dataset()
    print ("Making loaders...")
    train, val, test = make_loaders(inputs, gold)
    print ("Beginning to run model...")
    run_model(train, val, test)

if __name__ == "__main__":
    main()
