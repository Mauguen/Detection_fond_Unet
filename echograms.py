import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dask.array as da

class Echograms():
    """ISBI 2012 EM Cell dataset.
    """
    # '/home1/datawork/lmauguen/Data_fond/'
    def __init__(self, batch_size, root_dir=None, data_type='train', pad=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_type (string): either 'train' or 'validate'
        """
        self.root_dir = os.getcwd() if not root_dir else root_dir
        path = os.path.join(self.root_dir, 'data')
        self.data_type = data_type
        self.n_classes = 2
        self.batch_size = batch_size
        self.pad = pad
        
        if self.data_type == 'train':
            self.imgs_path = os.path.join(path, 'train-volume.h5')
        elif self.data_type == 'validate':
            self.imgs_path = os.path.join(path, 'validation-volume.h5')
            
        with h5py.File(self.imgs_path, 'r') as h5file:
            images = da.from_array(h5file['images'])
            labels = h5file['labels']
            self.n = images.shape[0]

            # Standard deviation and mean
            self.mean = images.mean().compute()
            self.std = images.std().compute()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """Returns a image sample from the dataset
        """
        with h5py.File(self.imgs_path, 'r') as h5file:
            ### Normalization
            images = (h5file['images'][idx : idx+self.batch_size]- self.mean) / self.std
            labels = h5file['labels'][idx : idx+self.batch_size]

            ### Padding and type torch.tensor
            new_imgs = []
            nb_train, nb_channels, _, _ = images.shape
            for i in range(nb_train):
                channel = []
                for j in range(nb_channels):
                    channel.append(np.pad(images[i][j], pad_width=self.pad, mode='reflect'))
                new_imgs.append(np.array(channel))
            images = np.array(new_imgs)

            batch = {'images': images, 'labels': labels}

        return batch

###############################################################################
# For testing
###############################################################################
def get_dataloader(batch_size):
    data = Echograms(batch_size)[0]
    print(data['images'].shape)
    return data


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
    data = Echograms(batch_size)
    n = len(data)
    for batch_start in range(0, n, batch_size):
        sample = data[batch_start]
        X = sample['images']
        y = sample['labels']
        print(X.shape, y.shape)
        # for i in range(X.shape[0]):
        #     image = X[i][0]
        #     mask = y[i][0]
        #     visualize(image, mask)

        break
