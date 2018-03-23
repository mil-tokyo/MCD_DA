import json
import os
import shutil

import torch
import sys


def set_debugger_org():
    if not sys.excepthook == sys.__excepthook__:
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_debugger_org_frc():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)


def set_trace():
    from IPython.core.debugger import Pdb
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def yes_no_input():
    while True:
        choice = raw_input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False


def check_if_done(filename):
    if os.path.exists(filename):
        print ("%s already exists. Is it O.K. to overwrite it and start this program?" % filename)
        if not yes_no_input():
            raise Exception("Please restart training after you set args.savename differently!")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_dic_to_json(dic, fn, verbose=True):
    with open(fn, "w") as f:
        json_str = json.dumps(dic, sort_keys=True, indent=4)
        if verbose:
            print (json_str)
        f.write(json_str)
    print ("param file '%s' was saved!" % fn)


def emphasize_str(string):
    print ('#' * 100)
    print (string)
    print ('#' * 100)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs, decay_epoch=15):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch == decay_epoch:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_class_weight_from_file(n_class, weight_filename=None, add_bg_loss=False):
    weight = torch.ones(n_class)
    if weight_filename:
        import pandas as pd

        loss_df = pd.read_csv(weight_filename)
        loss_df.sort_values("class_id", inplace=True)
        weight *= torch.FloatTensor(loss_df.weight.values)

    if not add_bg_loss:
        weight[n_class - 1] = 0  # Ignore background loss
    return weight
