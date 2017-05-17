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

NUM_TRAIN = 192
NUM_VAL = 16
BATCH_SIZE = 16
DATA_DIR = "../src/preprocess/prep_res"
PRINT_EVERY = 1

NUM_EPOCHS = 3
DROPOUT = 0.15
INIT_LR = 2e-3

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

        inputs, ground_truths = np.array(inputs), np.array(ground_truths)
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
            x_var = Variable(x.float().permute(0,3,1,2))
            y_var = Variable(y.float().permute(0,3,1,2))

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % PRINT_EVERY == 0:
                print('\tt = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def eval(model, dev_data, loss_fn):
    print ("Running evaluation...")
    total_loss = 0.0
    model.eval()
    for t, (x, y) in enumerate(dev_data):
        x_var = Variable(x.float().permute(0,3,1,2))
        y_var = Variable(y.float().permute(0,3,1,2))
        
        scores = model(x_var)
        print (scores.size())
        for i in range(scores.size()[0]):
            name = "./eval/{}_{}_".format(t, i)
            imsave(name + "gen.png", np.transpose(scores[i].data.numpy(), axes=[1,2,0]))
            imsave(name + "gold.png", np.transpose(y_var[i].data.numpy(), axes=[1,2,0]))
        
        total_loss += loss_fn(scores, y_var).data[0]

    print ("Total eval loss: %.4f, Avg eval loss: %.4f" % (total_loss, total_loss / NUM_VAL))


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
        nn.ReLU(inplace=True),
    )
    
    loss_fn = nn.L1Loss() # TODO: L2 loss
    optimizer = optim.Adam(model_base.parameters(), lr=INIT_LR)

    train(model_base, loss_fn, optimizer, train_data, num_epochs=NUM_EPOCHS) 
    eval(model_base, val_data, loss_fn)

def main():
    inputs, gold = load_dataset()
    train, val, test = make_loaders(inputs, gold)
    run_model(train, val, test)

if __name__ == "__main__":
    main()
