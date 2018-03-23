import numpy as np
from scipy.io import loadmat


def load_mnist(scale=True, usps=False, all_use=False):
    # Process MNIST
    if scale:
        mnist_train = np.reshape(np.load('/data/ugui0/ksaito/SVHN/train_mnist_32x32.npy'), (55000, 32, 32, 1))
        mnist_test = np.reshape(np.load('/data/ugui0/ksaito/SVHN/test_mnist_32x32.npy'), (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = np.load('/home/mil/ksaito/domain_adaptation/curriculum_da/MNIST_data/mnist_trainlabel.npy')
        mnist_labels_test = np.load('/home/mil/ksaito/domain_adaptation/curriculum_da/MNIST_data/mnist_testlabel.npy')
    else:
        mnist_train = np.load('../pytorch-da/MNIST_data/mnist_train.npy')
        mnist_test = np.load('../pytorch-da/MNIST_data/mnist_test.npy')
        mnist_labels_train = np.load('/home/mil/ksaito/domain_adaptation/curriculum_da/MNIST_data/mnist_trainlabel.npy')
        mnist_labels_test = np.load('/home/mil/ksaito/domain_adaptation/curriculum_da/MNIST_data/mnist_testlabel.npy')
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    # mnist_data = loadmat('data/mnist_data.mat')
    # if scale:
    #     mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
    #     mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
    #     mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
    #     mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
    #     mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
    #     mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
    #     mnist_labels_train = mnist_data['label_train']
    #     mnist_labels_test = mnist_data['label_test']
    # else:
    #     mnist_train = mnist_data['train_28']
    #     mnist_test =  mnist_data['test_28']
    #     mnist_labels_train = mnist_data['label_train']
    #     mnist_labels_test = mnist_data['label_test']
    #     mnist_train = mnist_train.astype(np.float32)
    #     mnist_test = mnist_test.astype(np.float32)
    #     mnist_train = mnist_train.transpose((0, 3, 1, 2))
    #     mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    if usps and all_use != 'yes':
        mnist_train = mnist_train[:2000]
        train_label = train_label[:2000]

    return mnist_train, train_label, mnist_test, test_label
