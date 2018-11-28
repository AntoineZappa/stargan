import torch
from torch.utils import data


class DatasetMix(data.Dataset):
    def __init__(self, source_dset, target_dset):
        self.target = target_dset
        self.source = source_dset

        self.nimages = min(len(self.target), len(self.source)) * 2

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        if idx < self.nimages / 2:
            image, mask, _ = self.source[idx]
            domain_label = torch.FloatTensor([True, False])
        else:
            idx = int(idx - (self.nimages / 2))
            image, mask, _ = self.target[idx]
            domain_label = torch.FloatTensor([False, True])

        return image, domain_label