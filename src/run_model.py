from __future__ import print_function, division

import glob
import os
import random
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.misc import imread, imsave
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, sampler

import loss_fns
from encodedecode import EncodeDecode
from normalizer import normalize, denorm
from viewmorphing import ViewMorphing

# Setup information
NUM_TRAIN = 20000   # 16000
NUM_VAL = 256
NUM_SAVED_SAMPLES = 8
BATCH_SIZE = 8
DATA_DIR = "preprocess/prep_res"
PRINT_EVERY = 20

# Default Model Hyperparams
NUM_EPOCHS = 15
DROPOUT = 0.15
INIT_LR = 2e-4

# Configuration Parameters
is_local = False
overfit_small = False
use_L2_loss = False
is_azure = True
NAME = "Translate_Learn_Decay"

if use_L2_loss:
    NAME = "_L2Loss"

if overfit_small:
    NUM_TRAIN = 64
    NUM_EPOCHS = 1500
    PRINT_EVERY = 1
    NAME +="_overfitting"

if os.path.exists("../john_local_flag.txt") or is_local:
    # this is because my local machine can't handle the batch size...
    is_local = True
    is_azure = False
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
    dtype = torch.FloatTensor
    NUM_TRAIN = 2
    NUM_VAL = 2
    NUM_SAVED_SAMPLES = 2  # needs to be less than or equal to batch size according to Matt
    torch.set_default_tensor_type("torch.FloatTensor")
else:
    dtype=torch.cuda.FloatTensor
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

curtime = strftime("_%m%d_%H%M%S", gmtime())
NAME = NAME + curtime
results_folder = "../../cs231n_results/" + NAME
os.makedirs(results_folder, exist_ok=True)


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


class RandomChunkSampler(ChunkSampler):
    def __iter__(self):
        iter_range = list(range(self.start, self.start + self.num_samples))
        random.shuffle(iter_range)
        return iter(iter_range)


def load_dataset():
    ground_truths = []
    inputs = []
    length = len(os.listdir(DATA_DIR))
    if os.path.isfile('saved_in_data.npy') and os.path.isfile('saved_ground_truths.npy'):
        print("Reading cached numpy data in from file...")
        inputs = np.load('saved_in_data.npy')
        ground_truths = np.load('saved_ground_truths.npy')
        print(len(inputs))
        return inputs, ground_truths

    for i, dir in enumerate(os.listdir(DATA_DIR)):
        print("\tOn dir %d of %d" % (i, length))
        src_p = DATA_DIR + "/" + dir
        if not os.path.isdir(src_p):
            continue
        src_f = sorted(glob.glob(src_p + "/*.jpg"))

        if not all(os.path.exists(f) for f in src_f): continue
        assert (len(src_f) % 3 == 0)

        files = zip(*[iter(src_f)]*3)
        if is_local:
            files = list(files)[:50]

        for zero, truth, one in files:
            t = imread(truth)
            z = imread(zero)
            o = imread(one)
            ground_truths.append(t)
            inputs.append(np.concatenate((z,o), axis=2))

    inputs, ground_truths = np.array(inputs), np.array(ground_truths)

    if is_azure:
        print("Not saving vars due to disk constraints")
    else:
        print("Caching numpy data for next run...")
        np.save('saved_in_data.npy', inputs)
        np.save('saved_ground_truths.npy', ground_truths)

    return inputs, ground_truths

def make_loaders(inputs, gold):
    inputs_t = torch.from_numpy(inputs).byte()
    gold_t = torch.from_numpy(gold).byte()
    dataset = TensorDataset(inputs_t, gold_t)

    offset = 15500 if overfit_small else 0

    train = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=RandomChunkSampler(NUM_TRAIN, 0+offset))
    val = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN+offset))
    test = None # For now

    return train, val, test

def train(model, loss_fn, optimizer, train_data, val_data, num_epochs = 1):
    losses=[]
    eval_losses=[]
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR * 10 ** -4)  # slow start (to prevent blowup
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d...' % (epoch + 1, num_epochs))
        model.train()
        if epoch == 6:
            print("Lowering rate for refinement")
            optimizer = optim.Adam(model.parameters(), lr=INIT_LR / 10)
        if epoch == 11:
            print("Lowering rate for refinement 2")
            optimizer = optim.Adam(model.parameters(), lr=INIT_LR / 100)
        for t, (x, y) in enumerate(train_data):
            if epoch == 0 and t == 50:
                optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
            # print(t)
            x_var = Variable(normalize(x).permute(0,3,1,2)).type(dtype)
            y_var = Variable(normalize(y).permute(0,3,1,2)).type(dtype)

            scores, oob_loss, _, _, _, _, _ = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % PRINT_EVERY == 0:
                norm_loss = calculate_norm_loss(x_var, y_var, scores, loss_fn)
                losses.append(norm_loss)
                print('\ttraining: t = %d, loss = %.4f, norm_loss= %.4f' % (t + 1, loss.data[0], norm_loss))
            if not is_local and t % (len(train_data) // 8) == 0 or overfit_small:
                eval_loss = evaluate(model, val_data, loss_fn)
                eval_losses.append(eval_loss)

            optimizer.zero_grad()
            (loss + oob_loss).backward()
            optimizer.step()

    os.makedirs("losses", exist_ok=True)
    np.save(results_folder + 'losses'+NAME, np.array(losses))
    np.save(results_folder +'losses/eval_losses'+NAME, np.array(eval_losses))

def calculate_norm_loss(x_var, y_var, pred_y, loss_fn):
    baseline_img = (x_var[:, :3,] + x_var[:, 3:])/2
    our_loss = loss_fn(pred_y, y_var).data[0]
    baseline_loss = loss_fn(baseline_img, y_var).data[0]
    return our_loss/ baseline_loss

def convert_and_save(name, img):
    """ Convert a gpu tensor into an image and save it"""
    imsave(name, np.transpose(denorm(img.data.cpu().numpy()), axes=[1,2,0]))

def evaluate(model, dev_data, loss_fn, save=False):
    print("Running evaluation...")

    model.eval()
    length = len(dev_data)

    # loss metrics
    l2_loss_fn = loss_fns.L2Loss()
    all_loss = []
    l2_losses = []
    for t, (x, y) in enumerate(dev_data):
        x_copy = np.copy(x.numpy())
        x_var = Variable(normalize(x).permute(0,3,1,2)).type(dtype)
        y_var = Variable(normalize(y).permute(0,3,1,2)).type(dtype)

        scores, _, C, M1, M2, res_img1, res_img2 = model(x_var)
        if (t >= length-2 and save):
            extra = results_folder + "extra/"
            os.makedirs(extra, exist_ok=True)
            for i in range(NUM_SAVED_SAMPLES):
                name = results_folder + "{}_{}_".format(t, i)
                convert_and_save(name + "gen.png", scores[i])
                convert_and_save(name + "gold.png", y_var[i])
                try:
                    convert_and_save(extra + "resgen1.png", res_img1[i])
                    convert_and_save(extra + "resgen2.png", res_img2[i])
                except Exception:
                    print(traceback.format_exc())

                # np.save(name + 'C', C.data.cpu().numpy())
                try:
                    np.save(extra + 'M1', M1.data.cpu().numpy())
                    np.save(extra + 'M2', M2.data.cpu().numpy())
                except Exception:
                    print(traceback.format_exc())
                # convert_and_save(name + "__Cx.png", )
                x_res = x_copy[i]
                try:
                    imsave(extra + "orig_0.png", x_res[:,:,:3])
                    imsave(extra + "orig_1.png", x_res[:,:,3:])
                except Exception:
                    print(traceback.format_exc())

        all_loss.append(calculate_norm_loss(x_var, y_var, scores, loss_fn))
        l2_losses.append(calculate_norm_loss(x_var, y_var, scores, l2_loss_fn))

    total_loss = sum(all_loss) / len(all_loss)
    total_l2_loss = sum(l2_losses) / len(l2_losses)
    print("Eval norm l2 loss: %.4f, norm total loss: %.4f" % (total_l2_loss, total_loss))
    return total_loss


def run_model(train_data, val_data, test_data):
    if False:
        model = nn.Sequential(
            EncodeDecode(),
            ViewMorphing()
        ).type(dtype)
    else:
        import translatelayer
        model = translatelayer.TranslateModel()

    #model = EncodeDecodeDirect().type(dtype)

    cur_loss_fn = loss_fns.TextureLoss2()
    if use_L2_loss:
        cur_loss_fn = loss_fns.L2Loss()
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)

    train(model, cur_loss_fn, optimizer, train_data, val_data, num_epochs=NUM_EPOCHS)

    try:
        torch.save(model, results_folder + 'model'+NAME+'.dat')
    except Exception:
        print(traceback.format_exc())

    try:
        if overfit_small:
            evaluate(model, train_data, cur_loss_fn, save=True)
        else:
            evaluate(model, val_data, cur_loss_fn, save=True)
    except Exception:
        print(traceback.format_exc())


def main():
    print("Loading dataset...")
    inputs, gold = load_dataset()
    print("Making loaders...")
    train_data, val, test = make_loaders(inputs, gold)
    print("Beginning to run model...")
    run_model(train_data, val, test)

if __name__ == "__main__":
    import logging
    logging.basicConfig(format='%(asctime)s    %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

    file_handler = logging.FileHandler(results_folder + "model_perf"+NAME + curtime+".log")
    file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s    %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(file_handler)
    print = logging.info
    try:
        main()
    except BaseException as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e
