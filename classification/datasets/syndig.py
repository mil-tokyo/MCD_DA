import numpy as np
from scipy.io import loadmat
import sys
sys.path.append('../utils/')
from utils.utils import dense_to_one_hot


def load_syndig():
    synth_train = loadmat('data/synth_train_32x32.mat')
    synth_test = loadmat('data/synth_test_32x32.mat')
    synth_train_im = synth_train['X']
    synth_train_im = synth_train_im.astype(np.float32)
    synth_train_im = synth_train_im.transpose(3, 0, 1, 2)
    synth_label_train = synth_train['y']
    synth_label_train = dense_to_one_hot(synth_label_train)
    synth_test_im = synth_test['X']
    synth_test_im = synth_test_im.astype(np.float32)
    synth_test_im = synth_test_im.transpose(3, 0, 1, 2)
    synth_label_test = synth_test['y']
    synth_label_test = dense_to_one_hot(synth_label_test)
    synth_train_im = synth_train_im.transpose(0, 3, 1, 2)
    synth_test_im = synth_test_im.transpose(0, 3, 1, 2)

    return synth_train_im, synth_label_train, synth_test_im, synth_label_test
