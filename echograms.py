import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dask.array as da
import torch
from torch.utils.data import DataLoader, Dataset
import random

class Echograms(Dataset):
    """ISBI 2012 EM Cell dataset.
    """
    # '/home1/datawork/lmauguen/Data_fond/'
    def __init__(self, freq='', root_dir=None, data_type='train', pad=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_type (string): either 'train' or 'validate'
        """
        self.root_dir = os.getcwd() if not root_dir else root_dir
        path = os.path.join(self.root_dir, f'data{freq}')
        self.data_type = data_type
        self.n_classes = 2
        self.pad = pad
        
        if self.data_type == 'train':
            self.imgs_path = os.path.join(path, 'train-volume.h5')
        elif self.data_type == 'validate':
            self.imgs_path = os.path.join(path, 'validation-volume.h5')
        elif self.data_type == 'param':
            self.imgs_path = os.path.join(path, 'parametrage-volume.h5')
            
        with h5py.File(self.imgs_path, 'r') as h5file:
            images = da.from_array(h5file['images'])
            labels = h5file['labels']
            self.n = images.shape[0]
            self.n_channels = images.shape[1]

            # Standard deviation and mean
            self.mean = images.mean().compute()
            self.std = images.std().compute()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.imgs_path, 'r') as h5file:
            ### Normalization
            image = (h5file['images'][idx]- self.mean) / self.std
            label = h5file['labels'][idx]

            ### Padding
            n_channels = self.n_channels
            channel = []
            for i in range(n_channels):
                channel.append(np.pad(image[i], pad_width=self.pad, mode='reflect'))
            image = np.array(channel)
            sample = {'image':image, 'label':label}

        return sample

###############################################################################
# For testing
###############################################################################
## Controlling sources of randomness
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(batch_size, num_worker):
    data = Echograms()
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                  )
    return train_dataloader


def visualize(image, mask):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')

    plt.show()


if __name__ == '__main__':
    batch_size = 5
    num_worker = 6
    dataloader = get_dataloader(batch_size, num_worker)

    for step, sample in enumerate(dataloader):
        X = sample['image']
        y = sample['label']
        print(X.shape)
        break
