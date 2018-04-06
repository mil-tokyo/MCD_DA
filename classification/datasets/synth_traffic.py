import numpy as np
import cPickle as pkl

def load_syntraffic():
    data_source = pkl.load(open('../data/data_synthetic'))
    source_train = np.random.permutation(len(data_source['image']))
    data_s_im = data_source['image'][source_train[:len(data_source['image'])], :, :, :]
    data_s_im_test = data_source['image'][source_train[len(data_source['image']) - 2000:], :, :, :]
    data_s_label = data_source['label'][source_train[:len(data_source['image'])]]
    data_s_label_test = data_source['label'][source_train[len(data_source['image']) - 2000:]]
    data_s_im = data_s_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_s_im_test = data_s_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_s_im, data_s_label, data_s_im_test, data_s_label_test