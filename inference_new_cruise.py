# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % inference_new_cruise.py
# % -------------------------------
# % Summary : script used to predict the bottom line on echograms with the CNN previously trained with Pytorch
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % OUTPUTS:
# % - bottom_line
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import random
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
from datetime import datetime
import h5py
import dask.array as da
from scipy.io import savemat, loadmat

from unet import UNet
import matplotlib.colors as colors


# Function to extract data from echogram files and save it into a new HDF5 file.
def extraction_1freq(echogram, freq, indice_freq, file_name, patchs_size):
    test = []
    cs, ds = [], []
    zero = 0
    with h5py.File(f'{echogram}Echogram.mat', 'r') as file2:
        # Sampling resolution
        res_echantillonnage = file2[f'Depth{freq}'][0, 0]
        nb_imgs = int(file2[f'Echogram{freq}'].shape[0] / patchs_size) + 1
        for i in range(nb_imgs):
            if i == nb_imgs-1:
                a, b = file2[f'Echogram{freq}'].shape[0] - patchs_size, file2[f'Echogram{freq}'].shape[0]
            else :
                a, b = zero, zero + patchs_size

            # Extracting necessary parameters
            # Bottoms
            Bottom = np.array(file2['Bottom'][a:b, indice_freq]).reshape((-1, 1))
            
            # Calculating median, max, min for image cropping
            med = int(np.nanmedian(Bottom) / res_echantillonnage)
            max_echo = file2[f'Echogram{freq}'].shape[1]
            # max_bottom = int(np.nanmax(Bottom) / res_echantillonnage)
            min_bottom = int(np.nanmin(Bottom) / res_echantillonnage)
            c, d = (med - (patchs_size // 2)), (med - (patchs_size // 2)) + patchs_size

            # Adjust cropping bounds
            if c < 0:
                c, d = 0, patchs_size
            elif d > max_echo:
                c, d = (max_echo - patchs_size), max_echo
            elif min_bottom < c:
                c, d = min_bottom - 10, min_bottom - 10 + patchs_size
            
            # Extracting the required sections
            Echogram = file2[f'Echogram{freq}'][a:b, c:d]
            Depth = np.array(file2[f'Depth{freq}'][:, c:d]).T
            Time_conv = np.array(file2['Time'][a:b], dtype=int).flatten()

            test.append(np.array(Echogram.T))
            cs.append(c)
            ds.append(d)
            zero += patchs_size
            print("Number of generated images: " + str(len(test)))
            
        extra_pings, pings_complete_patchs = file2[f'Echogram{freq}'].shape[0] - (int(file2[f'Echogram{freq}'].shape[0] / patchs_size) * patchs_size), (int(file2[f'Echogram{freq}'].shape[0] / patchs_size) * patchs_size)

    # Save extracted data into an HDF5 file
    with h5py.File(f'{echogram}imagettes.h5', 'w') as h5file:
        h5file.create_dataset('images', data=np.array(test).reshape((nb_imgs, 1, patchs_size, patchs_size)).astype('uint8'), compression="gzip", compression_opts=9, dtype='uint8')
        h5file.create_dataset('c', data=np.array(cs).reshape((nb_imgs)), compression="gzip", compression_opts=9)
        h5file.create_dataset('d', data=np.array(ds).reshape((nb_imgs)), compression="gzip", compression_opts=9)
    return extra_pings, pings_complete_patchs


# Dataset class for loading and processing echogram data.
class Echograms(Dataset):
    def __init__(self, root_dir, pad=0):
        """
        Args:
            root_dir (string): Directory containing the images.
            pad (integer) : Padding to apply to images.
        """
        self.root_dir = os.path.join(os.getcwd(), root_dir)
        self.n_classes = 1
        self.pad = pad

        # Load images as a Dask array for efficient processing
        with h5py.File(self.root_dir, 'r') as h5file:
            images = da.from_array(h5file['images'], chunks=(-1, -1, 100, 100))
            self.shape = images.shape
            self.n_channels = images.shape[1]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.root_dir, 'r') as h5file:
            # Normalize image
            image = h5file['images'][idx] / 255  # Convert uint8 data to [0,1]
            cs= h5file['c'][idx]
            ds = h5file['d'][idx]

            # Apply padding
            n_channels = self.n_channels
            channel = []
            for i in range(n_channels):
                channel.append(np.pad(image[i], pad_width=self.pad, mode='reflect'))
            image = np.array(channel)
            sample = {'image': image, 'indice': idx, 'c': cs, 'd': ds}

        return sample


# Function to parse command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predictions'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of workers to load data'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16, metavar='N',
        help='Input batch size for training (default: 16)'
    )
    args = parser.parse_args()
    return args


# Function to seed random number generators for reproducibility.
def seed_worker(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True


# Function to create a DataLoader for testing data.
def get_testloader(batch_size, num_worker, root_dir):
    data = Echograms(root_dir=root_dir)
    n = len(data)
    g = torch.Generator()
    g.manual_seed(0)
    test_dataloader = DataLoader(data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 worker_init_fn=seed_worker,
                                 pin_memory=True,
                                 generator=g
                                 )
    return test_dataloader, n


# Function to make predictions using the model.
def predict(im, model):
    y_pred = model(im)
    pred = torch.sigmoid(y_pred) > 0.5
    return pred


# Function to visualize the results including echogram, prediction, and ground truth.
def visualize(image, pred, depth, time, label=None):
    """Create visualizations"""
    n_plot = 2 if label is None else 4

    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    if n_plot > 2:
        fig = plt.figure()
        ax = fig.add_subplot(1, n_plot, 1)
        imgplot = plt.imshow(image)
        ax.set_title('Echogram')
        plt.colorbar(imgplot, orientation='horizontal')
        ax = fig.add_subplot(1, n_plot, 2)
        imgplot2 = plt.imshow(pred, cmap=cmap, norm=norm)
        ax.set_title('Prediction')
        plt.colorbar(imgplot2, orientation='horizontal', ticks=[0, 1])
        ax = fig.add_subplot(1, n_plot, 3)
        imgplot3 = plt.imshow(label, cmap=cmap, norm=norm)
        ax.set_title('Ground Truth')
        plt.colorbar(imgplot3, orientation='horizontal', ticks=[0, 1])
        ax = fig.add_subplot(1, n_plot, 4)
        imgplot4 = plt.imshow(pred - label, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar(imgplot4, orientation='horizontal', ticks=[-1, 0, 1])
        ax.set_title('Difference')
        fig.tight_layout()
    else:
        Time, Depth = np.meshgrid(time.reshape((-1,)), depth.reshape((-1,)))
        plt.figure()
        echogram = plt.pcolormesh(Time, Depth, image)
        plt.plot(time.reshape((-1,)), pred, label='Predicted Bottom Line')
        plt.legend()
        plt.colorbar(echogram)


# Function to create transparent colormaps for visualization.
def colormaps():
    red_colors = plt.cm.Reds(np.linspace(0, 1, 256))
    green_colors = plt.cm.Greens(np.linspace(0, 1, 256))
    for i in range(red_colors.shape[0]):
        alpha = i / (red_colors.shape[0] - 1)
        red_colors[i, 3] = alpha
        green_colors[i, 3] = alpha
    red_transparent = colors.LinearSegmentedColormap.from_list('RedTransparent', red_colors)
    green_transparent = colors.LinearSegmentedColormap.from_list('GreenTransparent', green_colors)
    return red_transparent, green_transparent

# Function to create final visualizations of the echogram and bottom line predictions.
def final_visu(echogram, freq, indice_freq, file_bottom_name):
    echograms = []
    width = 10000
    r = 4000
    with h5py.File(f'{echogram}Echogram.mat', 'r') as file:
        nb_imgs = int(file[f'Echogram{freq}'].shape[0] / width)
        for i in range(nb_imgs):
            a, b = i * width, (i + 1) * width

            # Extract sections
            Echogram = file[f'Echogram{freq}'][a:b, :r].T
            echograms.append(Echogram)

    CleanBottom = np.load(f'{file_bottom_name}', allow_pickle=True)
    for i in range(0, len(echograms)):
        plt.figure()
        plt.imshow(echograms[i])
        # plt.plot(times[i], CleanBottom[i * width: (i + 1) * width], color='r', alpha=0.5)
        plt.plot(CleanBottom[i * width: (i + 1) * width], color='r', alpha=0.5)
        image_path = f'{echogram}Images_fond_corrigé/echogram{i}'
        plt.savefig(image_path)
        # plt.show()
        plt.close()

# Main function to execute the script
if __name__ == '__main__':
    args = parse_args()
    display = True
    campagne = input('Campaign name to process: ')
    echogram = input("Location of the data? Example: D:/PFE/Detection_fond/Extraction_donnees/ ")
    echogram = echogram.replace("\\", "/")
    if not echogram.endswith('/'):
        echogram += '/'

    # Load frequency information from the file
    with h5py.File(f'{echogram}Echogram.mat', 'r') as file2:
        freqs = [key.split("Echogram", 1)[1] for key in [key for key in file2.keys() if "Echogram" in key]]
    freq = input(f"Working frequency? Available frequencies: {freqs} ")
    indice_freq = freqs.index(freq)
    file_name = f'{echogram}imagettes.h5'
    # with h5py.File(f'{echogram}Echogram.mat', 'r') as file2:
    #     length_echo = file2[f'Echogram{freq}'].shape[0]
    # patchs_size = closest_divisor_to_100(length_echo)
    # print(f'Taille des patchs : {patchs_size}')
    patchs_size = 100

    # Extract data for the selected frequency and save to HDF5
    extra_pings, pings_complete_patchs = extraction_1freq(echogram, freq, indice_freq, file_name, patchs_size)

    # Load test data
    dataloader, n = get_testloader(batch_size=args.batch_size, num_worker=args.num_workers, root_dir=file_name)

    # Load the pre-trained model
    args.model = 'UNet50_200_FAROFA_SCOPES_PIRATA.pt'
    checkpoint_path = os.path.join(os.getcwd(), f'models/{args.model}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(n_classes=1, in_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = (torch.device(f'cuda') if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device).double()
    model.eval()

    # Plot training and validation losses if specified
    courbes = False
    if courbes:
        print(np.argmin(np.array(checkpoint['train_loss']) + np.array(checkpoint['test_loss'])))
        nb_epochs = len(checkpoint['train_loss'])
        plt.figure()
        epoques = np.arange(start=1, stop=nb_epochs + 1, step=1)
        plt.subplot(1, 2, 1)
        plt.plot(epoques, checkpoint['train_loss'], label='Train loss')
        plt.plot(epoques, checkpoint['test_loss'], label='Validation loss')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title("Training and validation losses ")
        plt.subplot(1, 2, 2)
        plt.plot(epoques, checkpoint['metrics']['pix_acc'], label='Pixel accuracy during training')
        plt.plot(epoques, checkpoint['metrics']['IOU'], label='Intersection over union score')
        plt.xticks(np.arange(start=1, stop=nb_epochs + 1, step=int(nb_epochs / 10)))
        plt.legend()
        plt.title("Pixel accuracy during training and IOU score")
        plt.xlabel('Epochs')
        plt.show()

    # Create colormaps for visualization
    red_transparent, green_transparent = colormaps()

    t_ini = time.time()
    
    # Initialisation de bottom_line_pred_tot
    bottom_line_pred_tot = np.zeros((pings_complete_patchs + extra_pings, 1))
    
    # Indice de suivi pour le remplissage
    current_index = 0
    
    # Make predictions on each sample in the test set
    for step, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        indexs = sample['indice']
        cs = sample['c']
        ds = sample['d']
        preds = predict(images, model).cpu().detach().numpy()

        if (step * args.batch_size * 100 // n) % 1 == 0:
            print(f'Progress : {step * args.batch_size * 100 // n}%')
        for i in range(preds.shape[0]):
            pred = preds[i, 0].astype(int)
            image = images[i, 0].cpu().detach().numpy()
            c = cs[i].cpu().detach().numpy()
            
            # Vérification de la condition
            if current_index == pings_complete_patchs:
                bottom_line_pred = np.argmax(pred, axis=0)[-extra_pings:] + c
                bottom_line_pred_tot[current_index:current_index + extra_pings, 0] = bottom_line_pred
                current_index += extra_pings  # Mise à jour de l'indice
            else:
                bottom_line_pred = np.argmax(pred, axis=0) + c
                bottom_line_pred_tot[current_index:current_index + patchs_size, 0] = bottom_line_pred
                current_index += patchs_size
                # print(f'{current_index} / {pings_complete_patchs}')
                
    # Save and visualize the final results
    savemat(f'{echogram}CleanBottom_{campagne}_{freq}kHz.mat', {'bottom_line_pred_tot': np.array(bottom_line_pred_tot).flatten().reshape((1, -1))})
    np.save(file=f'{echogram}CleanBottom_{campagne}_{freq}kHz', arr=np.array(bottom_line_pred_tot).flatten())
    final_visu(echogram, freq, indice_freq, f'{echogram}CleanBottom_{campagne}_{freq}kHz.npy')

    t_fin = time.time()
    print(f'Execution time: {(t_fin - t_ini) / n:.2f} s/ping')
