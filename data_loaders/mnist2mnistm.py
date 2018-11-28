import numpy as np
import os
import struct
import torch
from PIL import Image
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision import transforms

from data_loaders.DatasetMix import DatasetMix


def mnist2mnistm(mode, image_dir, transform):
    d1 = MNIST(mode, image_dir, transform_list=[transform])
    d2 = MNIST_M(mode, image_dir, transform_list=[transform])
    d = DatasetMix(d1, d2)
    d.image_size = 32
    return d


class MNIST(Dataset):
    def __init__(self, split, path, transform_list=(), out_channels=3, mask_threshold=0):
        # A better mask threshold value is 0.7
        assert split in ['train', 'valid', 'test'], 'Not a valid split'

        # Paths
        if split == 'train':
            f_mnist_images = os.path.join(path, 'MNIST', 'raw', 'train-images-idx3-ubyte')
            f_mnist_labels = os.path.join(path, 'MNIST', 'raw', 'train-labels-idx1-ubyte')
        else:
            f_mnist_images = os.path.join(path, 'MNIST', 'raw', 't10k-images-idx3-ubyte')
            f_mnist_labels = os.path.join(path, 'MNIST', 'raw', 't10k-labels-idx1-ubyte')

        # Images
        with open(f_mnist_images, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
            self.images = np.pad(self.images, ((0, 0), (2, 2), (2, 2)), 'constant',
                                 constant_values=((0, 0), (0, 0), (0, 0)))
            if out_channels > 1:
                self.images = np.tile(self.images[:, :, :, np.newaxis], out_channels)

        # Labels
        with open(f_mnist_labels, 'rb') as f:
            struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, dtype=np.uint8)
        self.labels = torch.LongTensor(self.labels)

        self.transform_list = transform_list
        self.out_channels = out_channels
        self.n_classes = 2
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image
        image = self.images[idx]  # Range [0,255]

        # Mask
        if self.out_channels == 1:
            mask = torch.LongTensor((image > self.mask_threshold).astype(np.uint8))
        else:
            mask = torch.LongTensor((image[:, :, 0] > self.mask_threshold).astype(np.uint8))

        # Label
        label = self.labels[idx]

        image = Image.fromarray(image)
        for t in self.transform_list:
            image = t(image)
        # image = transforms.ToTensor()(image)  # Range [0,1]

        return image, mask, label


class MNIST_M(Dataset):
    def __init__(self, split, path, transform_list=()):
        assert split in ['train', 'valid', 'test'], 'Not a valid split'
        self.n_classes = 2

        # Paths to images and labels
        if split == 'train':
            self.path_images = os.path.join(path, 'MNIST_M', 'mnist_m_train')
            self.path_labels = os.path.join(path, 'MNIST_M', 'mnist_m_train_labels.txt')
            self.nimages = 59001
        else:
            self.path_images = os.path.join(path, 'MNIST_M', 'mnist_m_test')
            self.path_labels = os.path.join(path, 'MNIST_M', 'mnist_m_test_labels.txt')
            self.nimages = 9001

        # Labels
        self.labels = []
        with open(self.path_labels, 'r') as f:
            for line in f:
                self.labels.append(int(line.split(' ')[1]))
        self.labels = torch.LongTensor(self.labels)

        # Masks
        if split == 'train':
            path_images_mnist = os.path.join(path, 'MNIST', 'raw', 'train-images-idx3-ubyte')
        else:
            path_images_mnist = os.path.join(path, 'MNIST', 'raw', 't10k-images-idx3-ubyte')

        with open(path_images_mnist, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            self.masks = (np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols) > 0).astype(np.uint8)
            self.masks = pad(torch.LongTensor(self.masks), (2, 2, 2, 2), mode='constant', value=0)

        self.transform_list = transform_list
        self.pad2 = transforms.Pad(2)

    def __len__(self):
        return self.nimages

    def __getitem__(self, idx):
        # Image
        fname = '00000000' + str(idx) + '.png'
        fname = fname[-12:]
        image = Image.open(os.path.join(self.path_images, fname))  # Range [0,255]

        for t in self.transform_list:
            image = t(image)
        # image = transforms.ToTensor()(image)  # Range [0,1]

        # Mask
        mask = self.masks[idx]

        # Label
        label = self.labels[idx]

        return image, mask, label
