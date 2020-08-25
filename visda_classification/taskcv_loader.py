import random
import torch.utils.data
import torchvision.transforms as transforms
#import torchnet as tnt
# pip install future --upgrade
from builtins import object
from pdb import set_trace as st
import torch.utils.data as data_utils
class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size, flip):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size
        self.flip = flip

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            if self.flip and random.random() < 0.5:
                idx = [i for i in range(A.size(3) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(3, idx)
                B = B.index_select(3, idx)
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}

class CVDataLoader(object):
    def initialize(self, dataset_A,dataset_B,batch_size,shuffle=True):
        #normalize = transforms.Normalize(mean=mean_im,std=std_im)
        self.max_dataset_size = float("inf")
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4)
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4)
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False
        self.paired_data = PairedData(data_loader_A, data_loader_B, self.max_dataset_size, flip)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.max_dataset_size)
