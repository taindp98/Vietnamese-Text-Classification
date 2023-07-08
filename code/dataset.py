import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx][0].astype(np.float32)
        y = self.y[idx]
        return torch.from_numpy(x), y
