# -*- coding = utf-8 -*-
# @Time : 1/18/25 00:14
# @Author : Tracy
# @File : data_loader.py
# @Software : PyCharm

import torch
from torch.utils.data import Dataset

class FTDataset(Dataset):

    def __init__(self, data_series, seq_len):
        assert len(data_series) >= seq_len
        self.data_series = data_series
        self.seq_len = seq_len

    def __len__(self):
        return max(len(self.data_series)-self.seq_len, 0)

    def __getitem__(self, idx):
        return torch.tensor(self.data_series[idx:idx + self.seq_len]).reshape(-1, self.seq_len).to(torch.float32)

