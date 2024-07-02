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


from unet import UNet
import matplotlib.colors as colors
from echograms import Echograms

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet10_200_01.07.pt',
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
    # worker_seed = torch.initial_seed() % 2 ** 32
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def get_testloader(batch_size, num_worker, freq, data_used):
    data = Echograms(freq, data_type=data_used)
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
    pred = torch.argmax(y_pred, dim=1)
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
    plt.show()

def colormaps():
    # Récupération de la colormap Reds de matplotlib
    red_colors = plt.cm.Reds(np.linspace(0, 1, 256))
    green_colors = plt.cm.Greens(np.linspace(0, 1, 256))

    # Remplacement du blanc par du transparent
    for i in range(red_colors.shape[0]):
        # Calcul de l'alpha en fonction de la position dans la colormap
        alpha = i / (red_colors.shape[0] - 1)
        # Mélange entre transparent et la couleur rouge en fonction de l'alpha
        red_colors[i, 3] = alpha  # Définition de l'alpha pour créer l'effet de fondu
        green_colors[i, 3] = alpha

    # Création de la nouvelle colormap avec les couleurs modifiées
    red_transparent = colors.LinearSegmentedColormap.from_list('RedTransparent', red_colors)
    green_transparent = colors.LinearSegmentedColormap.from_list('GreenTransparent', green_colors)
    return red_transparent, green_transparent

if __name__ == '__main__':
    args = parse_args()
    exemple = True
    index_applied = 100
    freq_visu = 0
    data_used = 'train'
    if data_used == 'train':
        vertical_res = 0.196416        #m/pix, SCOPES
    else :
        vertical_res = 0.0247472  # m/pix, FAROFA3

    # Load model once
    checkpoint_path = os.path.join(os.getcwd(), f'models/{args.model}')
    checkpoint = torch.load(checkpoint_path)#, map_location=torch.device('cpu'))
    model = UNet(n_classes=2, in_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = (torch.device(f'cuda') if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device).double()
    model.eval()

    red_transparent, green_transparent = colormaps()

    t_ini = time.time()
    dataloader, n = get_testloader(batch_size=args.batch_size, num_worker=args.num_workers, freq='200', data_used=data_used)
    diff_ping = []
    errors_squared = []
    ligne_fond_non_detect = 0
    ligne_fond_tot = 0
    fond_pres = 0
    cumulative_conf_matrix = np.zeros((2, 2))
    for step, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        labels = sample['label'].to(device).squeeze(1).long()  # remove channel dimension
        indexs = sample['indice']
        preds = predict(images, model).cpu().detach().numpy()

        if index_applied in indexs and exemple:
            index = np.where(indexs.cpu().detach().numpy() == index_applied)[0]
            pred = preds[index][0]
            image = images[index][0, 0]
            label = labels[index][0]

            if np.any(np.unique(pred) != 0):
                plt.figure()
                plt.imshow(image.cpu().detach().numpy())
                plt.imshow(pred, cmap=red_transparent)
                # plt.imshow(label.cpu().detach().numpy(), alpha=0.5, cmap=green_transparent)
                plt.show()

                ## Confusion matrix
                conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
                ## Ping per ping difference
                diff = np.array(pred-label)
                diff2 = pred&label
                diff3 = pred&~label
                nb_elem_diff = (diff2 == 1).sum()
                ligne_fond_non_detect = (diff == -1).sum()
                for i in range(pred.shape[1]):
                    pred_non_fond = np.array(np.where(np.array(diff3[i])==1)).T
                    fond_reel = np.array(np.where(np.array(label[i])==1)).T
                    if len(pred_non_fond)!=0:
                        differences = np.subtract(pred_non_fond, fond_reel)
                        diff_ping.extend(differences)
                        errors_squared.extend(differences ** 2)
                moy = np.nanmean(np.array(diff_ping)) * vertical_res
                med = np.nanmedian(np.array(diff_ping)) * vertical_res
                std = np.nanstd(np.array(diff_ping)) * vertical_res
                rmse = np.sqrt(np.nanmean(errors_squared)) * vertical_res
                print(f'Matrice de confusion: {conf_matrix}')
                # disp = ConfusionMatrixDisplay(conf_matrix)
                # disp.plot()
                ## F1-score
                f1 = f1_score(label.flatten(), pred.flatten())
                print(f'F1-score: {f1}')
                # print(f'Pourcentage bien classifié = {(nb_elem_diff*100/(diff.shape[0]*diff.shape[1])):.2f}%')
                print(f'Ligne de fond non détecté : {ligne_fond_non_detect * 100 / np.array(label == 1).sum():.2f}%')
                print(f'Moyenne des écarts : {moy:.2f}m')
                print(f'Médiane des écarts : {med:.2f}m')
                print(f'Écart-type des écarts : {std:.2f}m')
                print(f'RMSE des écarts : {std:.2f}m')
                if args.visualize:
                    visualize(image[freq_visu], pred, label)
            else:
                print('No detection')
            break

        elif not exemple :      #tot jeu de donnees
            if (step*args.batch_size * 100 / n) % 1 == 0:
                print(f'Progress : {step * 100 // n}%')
            for i in range(preds.shape[0]):
                pred = preds[i]
                image = images[i]
                label = labels[i]
                if np.any(np.unique(pred) != 0):
                    fond_pres += 1
                    ## Confusion matrix
                    conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
                    cumulative_conf_matrix += conf_matrix
                    ## Ping per ping difference
                    diff = np.array(pred - label)
                    diff3 = pred & ~label
                    ligne_fond_non_detect += (diff == -1).sum()
                    ligne_fond_tot += (label == 1).sum()
                    for i in range(pred.shape[1]):
                        pred_non_fond = np.array(np.where(np.array(diff3[i]) == 1)).T
                        fond_reel = np.array(np.where(np.array(label[i]) == 1)).T
                        if len(pred_non_fond) != 0:
                            differences = np.subtract(pred_non_fond, fond_reel)
                            diff_ping.extend(differences)
                            errors_squared.extend(differences ** 2)

    if not exemple and fond_pres != 0 :
        moy = np.nanmean(np.array(diff_ping)) * vertical_res
        med = np.nanmedian(np.array(diff_ping)) * vertical_res
        std = np.nanstd(np.array(diff_ping)) * vertical_res
        rmse = np.sqrt(np.nanmean(errors_squared)) * vertical_res
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

    elif fond_pres == 0 and not exemple:
        print('No detection')

    t_fin = time.time()
    print(f'Execution time: {t_fin - t_ini:.2f} s')