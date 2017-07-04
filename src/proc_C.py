import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

path = 'full_eval/'

m1 = np.load(path + '0_0_M1.npy')
m2 = np.load(path + '0_0_M2.npy')

for i in range(1):
    m1i = np.transpose(m1[i], (1, 2, 0))[:, :, 0]
    m2i = np.transpose(m2[i], (1, 2, 0))[:, :, 0]

    im1 = imread(path + '0_' + str(i) + '_orig_0.png')
    im2 = imread(path + '0_' + str(i) + '_orig_1.png')

    imgen = imread(path + '0_' + str(i) + '_gen.png')

    plt.figure(1)
    plt.subplot(231)
    plt.imshow(m1i)
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(im1)
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(m2i)
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(im2)
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(imgen)
    plt.axis('off')

    plt.savefig(path + '0_' + str(i) + '_all.png')
