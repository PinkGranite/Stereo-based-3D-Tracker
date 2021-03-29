import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import *


class my_tracking_dataset(Dataset):
    def __init__(self, inputs, targets):
        super(my_tracking_dataset, self).__init__()
        self.inp = inputs
        self.tgt = targets
        self.len = len(inputs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # 这里是以字典的方式返回数据项
        disc = {}
        disc.update({'inputs': self.inp[idx], 'targets': self.tgt[0]})
        return disc
