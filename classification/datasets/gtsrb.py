import numpy as np
import cPickle as pkl

def load_gtsrb():
    data_target = pkl.load(open('../data/data_gtsrb'))
    target_train = np.random.permutation(len(data_target['image']))
    data_t_im = data_target['image'][target_train[:31367], :, :, :]
    data_t_im_test = data_target['image'][target_train[31367:], :, :, :]
    data_t_label = data_target['label'][target_train[:31367]] + 1
    data_t_label_test = data_target['label'][target_train[31367:]] + 1
    data_t_im = data_t_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_t_im_test = data_t_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_t_im, data_t_label, data_t_im_test, data_t_label_test
