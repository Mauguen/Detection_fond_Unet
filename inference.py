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

from unet import UNet
import matplotlib.colors as colors

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make segmentation predicitons'
    )
    parser.add_argument(
        '--model', type=str, default='UNet7.pt',
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

if __name__ == '__main__':
    args = parse_args()
    exemple = True
    
    ### Résultats sur 1 exemple
    if exemple :
        # index_applied = np.random.randint(0, 400)
        # print('Index = ', index_applied)
        index_applied = 0
        freq_visu = 0
        t_ini = time.time()
    
        # load images and labels
        label_on = True
        if label_on:
            path = os.getcwd() + '/data/train-volume.h5'
            with h5py.File(path, 'r') as h5file :
                images = h5file['images']
                labels = h5file['labels']
                image = images[index_applied]
                label = labels[index_applied][0]
        else:
            path = os.getcwd() + '/data/train-volume.h5'
            with h5py.File(path, 'r') as h5file :
                images = h5file['images']
                image = images[index_applied]
    
        # Load model
        checkpoint_path = os.getcwd() + f'/models/{args.model}'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = UNet(n_classes=2, in_channels=1)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # Prédiction
        pred = predict(image, model)[2:-2, 2:-2]
        t_fin = time.time()
        print('Execusion time : ' + str("{:.2g}".format(t_fin - t_ini)) + ' s')
    
        if args.visualize:
            if label_on:
                # visualize result
                visualize(image[freq_visu], pred, label)
                conf_matrix = confusion_matrix(label.flatten(), pred.flatten())
                print('Matrice de confusion : '+str(conf_matrix))
                disp = ConfusionMatrixDisplay(conf_matrix)
                disp.plot()
                f1_score = f1_score(label.flatten(), pred.flatten())
                print('F1-score : '+str(f1_score))
                print(np.max(np.array(pred.flatten()-label.flatten())))
            else:
                # visualize result
                visualize(image, pred)
    
    ### Résultats sur un jeu de donnée test
    else :
        freq_visu = 0
        t_ini = time.time()
    
        # load images and labels
        label_on = True
        if label_on:
            path = os.getcwd() + '/data/train-volume.h5'
            with h5py.File(path, 'r') as h5file :
                images = h5file['images']
                labels = h5file['labels']
                n, c, h, w = images.shape
                preds=np.zeros((n, h*w))
                ground_truths=np.zeros((n, h*w))
                for i in range(n):
                    image = images[i]
                    label = labels[i][0].T
                    # Load model
                    checkpoint_path = os.getcwd() + f'/models/{args.model}'
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                    model = UNet(n_classes=2, in_channels=1)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Prédiction
                    pred = predict(image, model)[2:-2, 2:-2]
                    preds[i] = pred.flatten().reshape((-1,))
                    ground_truths[i] = label.flatten().reshape((-1,))
                conf_matrix = confusion_matrix(ground_truths.flatten(), preds.flatten())
                disp = ConfusionMatrixDisplay(conf_matrix)
                disp.plot()
                f1_score = f1_score(ground_truths.flatten(), preds.flatten())
                print('F1-score : '+str(f1_score))
        else:
            path = os.getcwd() + '/data/train-volume.h5'
            with h5py.File(path, 'r') as h5file :
                images = h5file['images']
                preds=[]
                for i in range(images.shape[0]):
                    image = images[i]
                    # Load model
                    checkpoint_path = os.getcwd() + f'/models/{args.model}'
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                    model = UNet(n_classes=2, in_channels=1)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # Prédiction
                    pred = predict(image, model)
                    preds.append(pred)
                
        t_fin = time.time()
        print('Execusion time : ' + str(np.round((t_fin - t_ini), 2)) + ' s')
            
    plt.show()