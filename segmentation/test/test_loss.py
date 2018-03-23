# coding: utf-8
import sys
import unittest

import numpy as np
import torch
from torch.autograd import Variable

sys.path.append("../")
from loss import MySymkl2d, Symkl2d


def kld(p, q):
    """Calculates Kullback–Leibler divergence"""
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log(p / q), axis=(p.ndim - 1))


def jsd(p, q):
    """Calculates Jensen-Shannon Divergence"""
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * kld(p, m) + 0.5 * kld(q, m)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class TestKLD(unittest.TestCase):
    """test class for loss.py
    """

    def test_one_d(self):
        np1_one_dim = np.array([0.7, 0.2, 0.1])
        np2_one_dim = np.array([0.1, 0.8, 0.1])

        t1_one_dim = Variable(torch.FloatTensor(np1_one_dim))
        t2_one_dim = Variable(torch.FloatTensor(np2_one_dim))

        in1 = softmax(np1_one_dim)
        in2 = softmax(np2_one_dim)
        actual = 0.5 * (kld(in2, in1) + kld(in1, in2))

        mysymkl = MySymkl2d()
        nmlsymkl = Symkl2d(size_average=False)
        averaged_symkl = Symkl2d(size_average=True)

        pred_mysymkl = mysymkl(t1_one_dim, t2_one_dim).data[0]
        pred_nmlsymkl = nmlsymkl(t1_one_dim, t2_one_dim).data[0]
        pred_averaged_symkl = averaged_symkl(t1_one_dim, t2_one_dim).data[0]

        self.assertAlmostEqual(pred_mysymkl * len(np1_one_dim), actual)
        self.assertAlmostEqual(pred_nmlsymkl, actual)
        self.assertAlmostEqual(pred_averaged_symkl, pred_mysymkl)

    def test_four_d(self):
        batch_size, n_ch, w, h = 16, 3, 4, 5

        np1_four_dim = np.random.random([batch_size, n_ch, w, h])
        np2_four_dim = np.random.random([batch_size, n_ch, w, h])

        t1_one_dim = Variable(torch.FloatTensor(np1_four_dim))
        t2_one_dim = Variable(torch.FloatTensor(np2_four_dim))

        in1 = softmax(np1_four_dim)  # TODO: Need to be fixed
        in2 = softmax(np2_four_dim)  # TODO: Need to be fixed
        actual = 0.5 * (kld(in2, in1) + kld(in1, in2))  # TODO: Need to be fixed

        mysymkl = MySymkl2d()
        nmlsymkl = Symkl2d(size_average=False)
        averaged_symkl = Symkl2d(size_average=True)

        pred_mysymkl = mysymkl(t1_one_dim, t2_one_dim).data[0]
        pred_nmlsymkl = nmlsymkl(t1_one_dim, t2_one_dim).data[0]
        pred_symkl = averaged_symkl(t1_one_dim, t2_one_dim).data[0]

        # self.assertAlmostEqual(pred_mysymkl * 3, actual) # TODO: Need to be fixed
        # self.assertAlmostEqual(pred_nmlsymkl, actual) # TODO: Need to be fixed
        self.assertAlmostEqual(pred_symkl, pred_mysymkl)
        self.assertAlmostEqual(pred_symkl, pred_nmlsymkl / (batch_size * n_ch * w * h))


if __name__ == "__main__":
    unittest.main()
