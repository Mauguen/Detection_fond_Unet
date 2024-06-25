# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, BoundaryNorm
import h5py
import matplotlib
matplotlib.use('TkAgg')
from joblib import Parallel, delayed
from functools import partial


from unet import UNet
import matplotlib.colors as colors

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet10_200_24.06.pt',
        help='model to use for inference'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=True,
        help='visualize the inference result'
    )
    args = parser.parse_args()
    return args

def predict(image, model):
    """Make prediction on image"""
    nb_channels = image.shape[0]
    channels = []
    for i in range (nb_channels):
        im_channel = np.pad(image[i], pad_width=2, mode='reflect')
        channels.append(im_channel)
    channels = torch.tensor(np.array(channels), dtype=torch.float32)
    im = channels.view(1, *channels.shape)
    model.eval()
    y_pred = model(im)
    pred = torch.argmax(y_pred, dim=1)[0]
    return pred

def visualize(image, pred, label=None):
    """make visualization"""
    n_plot = 2 if label is None else 3
    # # Récupération de la colormap Reds de matplotlib
    # red_colors = plt.cm.Reds(np.linspace(0, 1, 256))
    #
    # # Remplacement du blanc par du transparent
    # for i in range(red_colors.shape[0]):
    #     # Calcul de l'alpha en fonction de la position dans la colormap
    #     alpha = i / (red_colors.shape[0] - 1)
    #     # Mélange entre transparent et la couleur rouge en fonction de l'alpha
    #     red_colors[i, 3] = alpha  # Définition de l'alpha pour créer l'effet de fondu
    #
    # # Création de la nouvelle colormap avec les couleurs modifiées
    # red_transparent = colors.LinearSegmentedColormap.from_list('RedTransparent', red_colors)

    # Définir une colormap discrète avec blanc et rouge
    colors = ['white', 'red']
    cmap = ListedColormap(colors)

    # Définir les limites pour chaque couleur
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(1, n_plot, 1)
    imgplot = plt.imshow(image, cmap='viridis')
    ax.set_title('Échogramme')
    plt.colorbar(imgplot, orientation='horizontal')
    ax = fig.add_subplot(1, n_plot, 2)
    imgplot2 = plt.imshow(pred, cmap=cmap, norm=norm)
    ax.set_title('Prediction')
    plt.colorbar(imgplot2, orientation='horizontal', ticks=[0,1])
    if n_plot > 2:
        ax = fig.add_subplot(1, n_plot, 3)
        imgplot3 = plt.imshow(label, cmap=cmap, norm=norm)
        ax.set_title('Ground Truth')
        plt.colorbar(imgplot3, orientation='horizontal', ticks=[0,1])
    fig.tight_layout()

def process_image(image, label, model):
    pred = predict(image, model)[2:-2, 2:-2]
    return pred.flatten(), label.flatten()

def process_batch(images_batch, labels_batch, model):
    process_image_with_model = partial(process_image, model=model)
    results = Parallel(n_jobs=-1)(delayed(process_image_with_model)(image, label) for image, label in zip(images_batch, labels_batch))
    return results

if __name__ == '__main__':
    args = parse_args()
    exemple = True
    label_on = True
    # path = os.getcwd() + '/data/train-volume.h5'
    # vertical_res = 0.196416        #m/pix, SCOPES
    path = os.getcwd() + '/data/test-volume.h5'
    vertical_res = 0.0247472  # m/pix, FAROFA3

    # Load model once
    checkpoint_path = os.path.join(os.getcwd(), f'models/{args.model}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(n_classes=2, in_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])

    if exemple:
        # index_applied = np.random.randint(0, 7000)
        # print('Index = ', index_applied)
        index_applied = 0
        freq_visu = 0
        t_ini = time.time()

        with h5py.File(path, 'r') as h5file:
            images = h5file['images']
            # fig = plt.figure(figsize=(15, 15))
            # for i in range(1+index_applied, 101):
            #     ax = fig.add_subplot(10, 10, i)
            #     ax.imshow(images[i][freq_visu])  # Exemple de données
            #     ax.set_title(f'Subplot {i}')
            # plt.tight_layout()
            # plt.show()

            if label_on:
                labels = h5file['labels']
                image = images[index_applied]
                label = labels[index_applied][0]
            else:
                image = images[index_applied]

        pred = predict(image, model)[2:-2, 2:-2]
        t_fin = time.time()
        print(f'Execution time: {t_fin - t_ini:.2g} s')
        if label_on:
            if np.any(np.unique(pred) != 0):
                ## Confusion matrix
                conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
                print(f'Matrice de confusion: {conf_matrix}')
                # disp = ConfusionMatrixDisplay(conf_matrix)
                # disp.plot()
                ## F1-score
                f1 = f1_score(label.flatten(), pred.flatten())
                print(f'F1-score: {f1}')
                ## Ping per ping difference
                diff = np.array(pred-label)
                diff2 = pred&label
                diff3 = pred&~label
                plt.figure()
                im = plt.imshow(diff3, cmap='Reds')
                plt.imshow(label, alpha=0.5)
                plt.colorbar(im)
                plt.show()
                nb_elem_diff = (diff2 == 1).sum()
                ligne_fond_non_detect = (diff == -1).sum()
                # print(f'Pourcentage bien classifié = {(nb_elem_diff*100/(diff.shape[0]*diff.shape[1])):.2f}%')
                print(f'Ligne de fond non détecté : {ligne_fond_non_detect*100/np.array(label == 1).sum():.2f}%')
                diff_ping = []
                errors_squared = []
                for i in range(pred.shape[1]):
                    pred_non_fond = np.array(np.where(np.array(diff3[i])==1)).T
                    fond_reel = np.array(np.where(np.array(label[i])==1)).T
                    if len(pred_non_fond)!=0:
                        differences = np.subtract(pred_non_fond, fond_reel)
                        diff_ping.extend(differences)
                        errors_squared.extend(differences ** 2)
                moy = np.nanmean(np.array(diff_ping)) * vertical_res
                print(f'Moyenne des écarts : {moy:.2f}m')
                med = np.nanmedian(np.array(diff_ping)) * vertical_res
                print(f'Médiane des écarts : {med:.2f}m')
                std = np.nanstd(np.array(diff_ping)) * vertical_res
                print(f'Écart-type des écarts : {std:.2f}m')
                rmse = np.sqrt(np.nanmean(errors_squared)) * vertical_res
                print(f'RMSE des écarts : {std:.2f}m')
            else:
                print('No detection')
        if args.visualize and label_on:
            visualize(image[freq_visu], pred, label)
        elif args.visualize :
            visualize(image, pred)
    else:
        batch_size = 100
        t_ini = time.time()
        with h5py.File(path, 'r') as h5file:
            images = h5file['images'][:]
            if label_on:
                labels = h5file['labels']
                n, c, h, w = images.shape
                cumulative_conf_matrix = np.zeros((2,2))
                print(f"Loaded {n} images with shape ({c}, {h}, {w})")

                diff_ping = []
                errors_squared = []
                ligne_fond_non_detect = 0
                ligne_fond_tot = 0
                fond_pres = 0
                for i in range(0, n, batch_size):
                    print(f'Progress : {i*100//n}%')
                    images_batch = images[i:i + batch_size]
                    labels_batch = labels[i:i + batch_size]
                    results = process_batch(images_batch, labels_batch, model)

                    for pred, ground_truth in results:
                        if np.any(np.unique(pred) != 0):
                            fond_pres +=1
                            conf_matrix = confusion_matrix(ground_truth, pred)      #, labels=list(range(h * w))
                            cumulative_conf_matrix += conf_matrix
                            pred, ground_truth = pred.reshape((h, w)), ground_truth.reshape((h, w))
                            diff = np.array(pred - ground_truth)
                            # diff2 = pred & ground_truth
                            diff3 = np.array(pred & ~ground_truth)
                            ligne_fond_non_detect += np.sum(diff == -1)
                            ligne_fond_tot += np.sum(ground_truth == 1)
                            for i in range(w):
                                # print(np.array(np.where(diff3[i]==1)).shape, np.array(np.where(ground_truth[i]==1)).shape)
                                pred_non_fond = np.array(np.where(diff3[i] == 1)).T
                                fond_reel = np.array(np.where(np.array(ground_truth[i]) == 1)).T
                                if len(fond_reel) == 1 and len(pred_non_fond) != 0 :
                                    differences = np.subtract(pred_non_fond, fond_reel)
                                    diff_ping.extend(differences)
                                    errors_squared.extend(differences ** 2)
                if fond_pres != 0:
                    print(f'Ligne de fond non détectée : {ligne_fond_non_detect * 100 / ligne_fond_tot:.2f}%')
                    moy = np.nanmean(np.array(diff_ping, dtype=np.float32)) * vertical_res
                    print(f'Moyenne des écarts : {moy:.2f}m')
                    med = np.nanmedian(np.array(diff_ping, dtype=np.float32)) * vertical_res
                    print(f'Médiane des écarts : {med:.2f}m')
                    std = np.nanstd(np.array(diff_ping, dtype=np.float32)) * vertical_res
                    print(f'Écart-type des écarts : {std:.2f}m')
                    rmse = np.sqrt(np.nanmean(errors_squared, dtype=np.float32)) * vertical_res
                    print(f'RMSE des écarts : {std:.2f}m')
                    # disp = ConfusionMatrixDisplay(cumulative_conf_matrix)
                    # disp.plot()
                    print(f'Matrice de confusion: {cumulative_conf_matrix * 100 / np.sum(cumulative_conf_matrix)}')
                    precision = cumulative_conf_matrix[1, 1] / (cumulative_conf_matrix[1, 1] + cumulative_conf_matrix[1, 0])
                    recall = cumulative_conf_matrix[1, 1] / (cumulative_conf_matrix[1, 1] + cumulative_conf_matrix[0, 1])
                    # Calcul du F1-score
                    f1 = 2 * (precision * recall) / (precision + recall)
                    print(f'F1-score: {f1}')
                else :
                    print('No detection')
            else:
                preds = []
                for i in range(images.shape[0]):
                    image = images[i]
                    pred = predict(image, model)
                    preds.append(pred)

        t_fin = time.time()
        print(f'Execution time: {t_fin - t_ini:.2f} s')

    plt.show()
