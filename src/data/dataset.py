import torch
from torch.utils.data import Dataset
import numpy as np

class SeismicDataset(Dataset):
    def __init__(self, data_files, labels, transform=None):
        self.data_files = data_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), label
