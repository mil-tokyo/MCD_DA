import numpy as np
import gzip
import cPickle


def load_usps(all_use=False):
    f = gzip.open('data/usps_28x28.pkl', 'rb')
    data_set = cPickle.load(f)
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    if all_use == 'yes':
        img_train = img_train[inds][:6562]
        label_train = label_train[inds][:6562]
    else:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))
    return img_train, label_train, img_test, label_test
