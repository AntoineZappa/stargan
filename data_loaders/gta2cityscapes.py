import os
from PIL import Image
from torch.utils.data import Dataset

from data_loaders.DatasetMix import DatasetMix


def gta2cityscapes(image_dir, transform):
    d1 = UnpackedDataset(os.path.join(image_dir, 'GTAV', 'images'), transform)  # 3, 1052, 1914]
    d2 = UnpackedDataset(os.path.join(image_dir, 'cityscapes', 'leftImg8bit'), transform)  # 3, 1024, 2048
    d = DatasetMix(d1, d2, mix_mode='balanced')
    return d


class UnpackedDataset(Dataset):
    def __init__(self, path, transform=None):
        print('==> Dataset: {}'.format(self.__class__.__name__))
        self.transform = transform

        # Check paths
        if not os.path.exists(path):
            raise FileNotFoundError(path + ' not found.')

        # Tree search of the images
        print('Loading files in {}'.format(path))
        self.filenames = []
        for root, dirs, files in os.walk(path):
            # Filter file names
            l = list(filter(lambda x: x.endswith('.png'), files))

            # Generate path of the file
            l = list(map(lambda x: (x, os.path.join(root, x)), l))

            self.filenames.extend(l)

        print('Number of images: {}'.format(len(self.filenames)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx][1])
        if self.transform is not None:
            image = self.transform(image)
        return image
