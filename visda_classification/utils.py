import numpy as np
import torch
from numpy.random import *
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d, BatchNorm3d


def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines

def adjust_learning_rate(optimizer, epoch,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * 0.99#min(1, 2 - epoch/float(20))#0.95 best
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

