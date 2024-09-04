# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % echograms.py
# % -------------------------------
# % Summary : classe Echogram used to load samples with Pytorch's DataLoader
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % INPUTS:
# % - root_dir : folder containing the data to train, validate or test the CNN
# % - data_type : usage of data thus loaded
# % OUTPUTS:
# % - dataloader
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dask.array as da
import torch
from torch.utils.data import DataLoader, Dataset
import random


class Echograms(Dataset):
    def __init__(self, root_dir='D:/PFE/Codes_finaux/data/', data_type='train', pad=0):
        """
        Initialize the Echograms dataset.

        Args:
            root_dir (string): Directory with all the images.
            data_type (string): either 'train', 'validate', 'param' or 'test'.
            pad (integer): Padding size to apply to the images.
        """
        # Set the root directory for the data
        self.root_dir = os.getcwd() if not root_dir else root_dir
        path = self.root_dir

        # Number of classes in the dataset (assuming binary classification)
        self.n_classes = 1
        self.pad = pad

        # Select the appropriate dataset file based on data_type
        if data_type == 'train':
            self.imgs_path = os.path.join(path, 'train-volume_FAROFA_SCOPES.h5')
        elif data_type == 'validate':
            self.imgs_path = os.path.join(path, 'validation-volume_only_bottom_time.h5')
        elif data_type == 'param':
            self.imgs_path = os.path.join(path, 'param-volume_FAROFA3.h5')
        elif data_type == 'test':
            self.imgs_path = os.path.join(path, 'test-volume_FAROFA_70kHz.h5')

        # Load the image dataset and get the shape and number of channels
        with h5py.File(self.imgs_path, 'r') as h5file:
            images = da.from_array(h5file['images'], chunks=(-1, -1, 100, 100))
            self.shape = images.shape
            self.n_channels = images.shape[1]

    def __len__(self):
        # Return the number of samples in the dataset
        return self.shape[0]

    def __getitem__(self, idx):
        # Load a specific sample by index
        with h5py.File(self.imgs_path, 'r') as h5file:
            ### Normalize the image by dividing pixel values by 255 (assuming uint8 format)
            image = h5file['images'][idx] / 255
            label = h5file['labels'][idx]
            c, h, w = self.shape[1], self.shape[2], self.shape[3]

            ### Apply padding to the image
            n_channels = self.n_channels
            channel = []
            for i in range(n_channels):
                channel.append(np.pad(image[i], pad_width=self.pad, mode='reflect'))
            image = np.array(channel)

            # Return the sample as a dictionary
            sample = {'image': image, 'label': label, 'indice': idx, 'c': c, 'h': h, 'w': w}

        return sample


###############################################################################
# For testing
###############################################################################
## Controlling sources of randomness
def seed_worker(worker_id):
    """
    Seed the worker to ensure reproducibility in data loading.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(batch_size, num_worker):
    """
    Create a DataLoader for the Echograms dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_worker (int): Number of worker threads for loading data.

    Returns:
        DataLoader: Pytorch DataLoader object.
    """
    data = Echograms()
    print(len(data))
    g = torch.Generator()
    g.manual_seed(0)  # Ensure the same random sequence is used every time

    # Create the DataLoader
    train_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                  )
    return train_dataloader


def visualize(image, mask):
    """
    Visualize an image and its corresponding label (mask).

    Args:
        image (numpy array): Image data.
        mask (numpy array): Corresponding label data.
    """
    fig = plt.figure()

    # Plot the image
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Image')

    # Plot the label
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')

    plt.show()


if __name__ == '__main__':
    # Set batch size and number of workers for data loading
    batch_size = 5
    num_worker = 6

    # Get the DataLoader for the dataset
    dataloader = get_dataloader(batch_size, num_worker)

    no_bottom = 0
    # Iterate through batches of data
    for step, sample in enumerate(dataloader):
        X = sample['image']  # Batch of images
        y = sample['label']  # Batch of labels

        # Check for samples with no bottom detected
        for i in range(X.shape[0]):
            if np.all(np.array(y[i, 0]) == 0):
                no_bottom += 1
        print(step * batch_size, no_bottom)

    # Print the total number of samples with no bottom detected
    print(no_bottom)