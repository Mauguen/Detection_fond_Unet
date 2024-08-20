# -*- coding: utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import time
import random
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime

from unet import UNet
import matplotlib.colors as colors
from echograms import Echograms

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet32_200.pt',
        help='model to use for inference'
    )
    parser.add_argument(
        '--visualize', action='store_true', default=True,
        help='visualize the inference result'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of workers to load data'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16, metavar='N',
        help='input batch size for training (default: 3)'
    )
    args = parser.parse_args()
    return args

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

def get_testloader(batch_size, num_worker, data_used):
    data = Echograms(data_type=data_used, root_dir='D:/PFE/Codes_finaux/data/')
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

def predict(im, model):
    y_pred = model(im)
    pred = torch.sigmoid(y_pred) > 0.5
    return pred

def visualize(image, pred, label=None, time=False):
    """make visualization"""
    n_plot = 2 if label is None else 4

    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = fig.add_subplot(1, n_plot, 1)
    imgplot = plt.imshow(image)
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
        plt.colorbar(imgplot3, orientation='horizontal', ticks=[0, 1])
        ax = fig.add_subplot(1, n_plot, 4)
        imgplot4 = plt.imshow(pred - label, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar(imgplot4, orientation='horizontal', ticks=[-1, 0, 1])
        ax.set_title('Difference')
    fig.tight_layout()

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

if __name__ == '__main__':
    args = parse_args()
    exemple = False
    index_applied = 955
    data_used = 'train'
    if data_used == 'train' or 'validate':
        vertical_res = 0.196  #m/pix, SCOPES
    else :
        vertical_res = 0.024  # m/pix, FAROFA3

    ### Load model
    args.model = 'UNet50_200_FAROFA_SCOPES_PIRATA.pt'
    checkpoint_path = os.path.join(os.getcwd(), f'models/{args.model}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = UNet(n_classes=1, in_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = (torch.device(f'cuda') if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device).double()
    model.eval()

    ### Display of training and validation losses during training
    courbes = False
    if courbes:
        print(np.argmin(np.array(checkpoint['train_loss']) + np.array(checkpoint['test_loss'])))
        nb_epochs = len(checkpoint['train_loss'])
        plt.figure()
        epoques = np.arange(start=1, stop=nb_epochs+1, step=1)
        plt.subplot(1, 2, 1)
        plt.plot(epoques, checkpoint['train_loss'], label='Train loss')
        plt.plot(epoques, checkpoint['test_loss'], label='Validation loss')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title("Training and validation losses ")
        plt.subplot(1, 2, 2)
        plt.plot(epoques, checkpoint['metrics']['pix_acc'], label='Pixel accuracy durin training')
        plt.plot(epoques, checkpoint['metrics']['IOU'], label='Intersection over union score')
        plt.xticks(np.arange(start=1, stop=nb_epochs+1, step=int(nb_epochs/10)))
        plt.legend()
        plt.title("Pixel accuracy during training and IOU score")
        plt.xlabel('Epochs')
        plt.show()

    red_transparent, green_transparent = colormaps()

    t_ini = time.time()
    dataloader, n = get_testloader(batch_size=args.batch_size, num_worker=args.num_workers, data_used=data_used)

    ### Creation of variables used for the evaluation of the model
    diff_ping = []
    errors_squared = []
    bottom_line_pred_tot = []
    fond_non_detect = 0
    fond_tot = 0
    fond_pres = 0
    err_sup_1 = 0
    cumulative_conf_matrix = np.zeros((2, 2))

    ### Predictions on each sample of the test set
    for step, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        labels = sample['label'].to(device)
        indexs = sample['indice']
        preds = predict(images, model).cpu().detach().numpy()

        if index_applied in indexs and exemple:
            index = np.where(indexs.cpu().detach().numpy() == index_applied)[0]
            pred = preds[index][0, 0]
            image = images[index][0, 0].cpu().detach().numpy()
            label = labels[index][0, 0].cpu().detach().numpy()

            ## Ping per ping difference
            bottom_line = np.argmax(label, axis=0)
            bottom_line_pred = np.argmax(pred, axis=0)
            diff_bottom_line = np.abs(bottom_line - bottom_line_pred)
            std_line = np.std(diff_bottom_line) * vertical_res
            mean_line = np.mean(diff_bottom_line) * vertical_res
            diff_ping.append(diff_bottom_line)
            ## Confusion matrix
            conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
            cumulative_conf_matrix += conf_matrix
            ## Pixel per pixel difference
            diff = np.abs(np.array(pred - label))
            fond_non_detect += (diff == 1).sum(axis=1)
            fond_non_detect_img = (diff == 1).sum(axis=1)
            fond_tot += (label == 1).sum()
            std_img = np.std(fond_non_detect_img) * vertical_res
            mean_img = np.mean(fond_non_detect_img) * vertical_res
            nb_diff_img = np.sum(fond_non_detect_img)
            visualize(image, pred, label)
            plt.show()
            break

        elif not exemple :
            if (step*args.batch_size * 100 // n) % 1 == 0:
                print(f'Progress : {step*args.batch_size * 100 // n}%')
            for i in range(preds.shape[0]):
                pred = preds[i, 0].astype(int)
                image = images[i, 0].cpu().detach().numpy()
                label = labels[i, 0].cpu().detach().numpy()
                ## Line difference
                bottom_line = np.argmax(label, axis=0)
                bottom_line_pred = np.argmax(pred, axis=0)
                bottom_line_pred_tot.append(bottom_line_pred)
                diff_bottom_line = np.abs(bottom_line - bottom_line_pred)
                std_line = np.std(diff_bottom_line) * vertical_res
                mean_line = np.mean(diff_bottom_line) * vertical_res
                diff_ping.append(diff_bottom_line)
                ## Confusion matrix
                conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
                cumulative_conf_matrix += conf_matrix
                ## Ping difference
                diff = np.abs(np.array(pred - label))
                fond_non_detect += (diff == 1).sum(axis=1)
                fond_non_detect_img = (diff == 1).sum(axis=1)
                fond_tot += (label == 1).sum()
                ## Display of predictions fro which the difference between predicted and corrected bottom lines is over 1 pixel
                if mean_line > 1 :
                    print(f'Indice : {i+step*args.batch_size}')
                    visualize(image, pred, label)
                    plt.show()

    ### Evaluation of the model
    if not exemple:
        print(f'{err_sup_1} / {n}')
        moy = np.nanmean(fond_non_detect)/n
        med = np.nanmedian(fond_non_detect)/n
        std = np.nanstd(fond_non_detect)/n
        moy_line = np.nanmean(diff_ping)
        med_line = np.nanmedian(diff_ping)
        std_line = np.nanstd(diff_ping)
        print(f'Mauvaises detections : {np.sum(fond_non_detect)} / {fond_tot} soit {np.sum(fond_non_detect)*100/fond_tot:.2f}%')
        print(f'Moyenne des écarts de pixels : {moy:.2f}')
        print(f'Médiane des écarts de pixels : {med:.2f}')
        print(f'Écart-type des écarts de pixels : {std:.2f}')
        print(f'Moyenne des écarts de distance entre lignes de fond : {moy_line:.2f}pixels')
        print(f'Médiane des écarts de distance entre lignes de fond : {med_line:.2f}pixels')
        print(f'Écart-type des écarts de distance entre lignes de fond : {std_line:.2f}pixels')
        disp = ConfusionMatrixDisplay(cumulative_conf_matrix)
        disp.plot()
        plt.show()
        np.set_printoptions(precision=2, suppress=True)
        print(f'Matrice de confusion en pourcentages: {cumulative_conf_matrix * 100 / np.sum(cumulative_conf_matrix)}')
        precision = cumulative_conf_matrix[1, 1] / (cumulative_conf_matrix[1, 1] + cumulative_conf_matrix[1, 0])
        recall = cumulative_conf_matrix[1, 1] / (cumulative_conf_matrix[1, 1] + cumulative_conf_matrix[0, 1])
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'F1-score: {f1}')
        print(np.array(bottom_line_pred_tot))

    t_fin = time.time()
    print(f'Execution time: {(t_fin - t_ini)/n:.2f} s/ping')