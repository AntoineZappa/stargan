import torch
from torch.utils import data
import numpy as np


class DatasetMix(data.Dataset):

    # - With mix_mode='min' the longest dataset will be cut into the length of the shortest one. Ignoring some images.
    # - With mix_mode='complete' all the images will be taken into account. But, depending on the length of each dataset
    # problems of unbalance might appear.
    # - With mix_mode='balanced' all the images will be taken into account with balanced batches of images. But, images
    # of the shortest dataset will have multiple indexes.
    def __init__(self, dset1, dset2, mix_mode='min'):
        if len(dset1) < len(dset2):
            self.dset_short = dset1
            self.dset_long = dset2
        else:
            self.dset_short = dset2
            self.dset_long = dset1
        self.mix_mode = mix_mode

        if self.mix_mode == 'min':
            self.nimages = min(len(self.dset_long), len(self.dset_short)) * 2
        elif self.mix_mode == 'complete':
            self.nimages = len(self.dset_long) + len(self.dset_short)
        elif self.mix_mode == 'balanced':
            self.nimages = max(len(self.dset_long), len(self.dset_short)) * 2
        else:
            raise Exception("mix_mode='{}' not valid.".format(self.mix_mode))

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        if (self.mix_mode == 'complete' and idx < len(self.dset_short)) \
                or (self.mix_mode != 'complete' and idx < self.nimages / 2):
            if self.mix_mode == 'balanced':
                idx = idx % len(self.dset_short)
            image = self.dset_short[idx]
            domain_label = torch.FloatTensor([True, False])
        else:
            if self.mix_mode == 'complete':
                idx -= len(self.dset_short)
            else:
                idx = int(idx - (self.nimages / 2))
            image = self.dset_long[idx]
            domain_label = torch.FloatTensor([False, True])

        return image, domain_label
