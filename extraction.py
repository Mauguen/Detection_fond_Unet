# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % extraction.py
# % -------------------------------
# % Summary : script to extract the samples used to train, validate and test the CNN
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

import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import matplotlib.dates as mdates
import tifffile
import matplotlib.colors as colors
import os
import gc
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

############################### Version 3 - Images size from depth ###############################
### Fonctions annexes
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

def nb_ping_modif(filtering, echogram, echograms, figure=False):
    # Ouverture du fichier MAT en mode lecture seule
    with h5py.File(filtering, 'r') as file:
        with h5py.File(echogram, 'r') as file2:
            nb_ping, nb_freq = file['CleanBottom'].shape
            for i in range(nb_freq):
                print(f'Résolution verticale : {file2[depths[i]][0, 0]}')
                print(f'Portée : {file2[echograms[i]].shape[1] * file2[depths[i]][0, 0]}')
    return nb_ping

def affichage_echogram(time, bottom, cleanBottom, depth, Z, title):
    X, Y = np.meshgrid(time.squeeze(), depth.squeeze())

    plt.figure()
    plt.plot(time, -bottom, label='Fond initial')
    plt.plot(time, -cleanBottom, label='Fond corrigé')
    im = plt.pcolor(X, -Y, Z)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.colorbar(im, label='Sv (dB)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Profondeur (m)')
    plt.title(title)

### Extraction d'information et prétraitements
def extraction_1freq(patchs_size, phase, zero, filtering, echogram, echograms, depths, athwart, along, masks, indice_freq, test_echo=False):
    # Ouverture du fichier MAT en mode lecture seule
    with h5py.File(filtering, 'r') as file:
        with h5py.File(echogram, 'r') as file2:
            res_echantillonnage = file2[depths[indice_freq]][0, 0]
            a, b = zero, zero+patchs_size
            Depth = np.array(file2[depths[indice_freq]]).T
            Bottom = np.array(file2['Bottom'][a:b, indice_freq]).reshape((-1, 1))
            CleanBottom = np.array(file['CleanBottom'][a:b, indice_freq]).reshape((-1, 1))
            if test_echo:
                med = int(np.nanmedian(CleanBottom) / res_echantillonnage)
            else :
                med = int(np.nanmedian(Bottom) / res_echantillonnage)
            max = file2[echograms[indice_freq]].shape[1]
            max_bottom = int(np.nanmax(Bottom) / res_echantillonnage)
            min_bottom = int(np.nanmin(Bottom) / res_echantillonnage)
            c, d = (med - (patchs_size // 2)), (med - (patchs_size // 2)) + patchs_size
            if c < 0:
                c, d = 0, patchs_size
            elif d > max:
                c, d = (max - patchs_size), max
            elif min_bottom < c:
                c, d = min_bottom-10, min_bottom - 10 + patchs_size

            Mask_Cleaning = file[masks[indice_freq]][a:b, c:d]
            Echogram = file2[echograms[indice_freq]][a:b, c:d]

    return CleanBottom, Bottom, Depth, Echogram, Mask_Cleaning, res_echantillonnage, c, d, max

def filtre_bottom_1echogram(depth, echogram, cleanbottom, res_echantillonnage, c, d):
    mat_clean = np.zeros(echogram.shape)
    for i in range(echogram.shape[0]):
        indice = int(cleanbottom.flatten()[i]/res_echantillonnage)-1
        if indice <= d and indice >= c:
            mat_clean[i, int(indice-c):]=np.ones(mat_clean[i, int(indice-c):].shape)
    return mat_clean.T

### Création d'un set d'images en fonction du type d'extraction souhaité
def creation_imgs_fond_train_1_freq(patchs_size, phase, freq, nb_train, zero_ini, filtering, echogram, echograms, depths, athwart, along, masks, train_name):
    labels, train = [], []
    zero = zero_ini
    nb_imgs = 0
    no_bottom = 0
    imgs_fond = -1
    red_transparent, green_transparent = colormaps()
    for i in range (nb_train):
        if phase :
            CleanBottom, Bottom, Depth, Echogram, Phase_along, Phase_athwart, Mask_Cleaning, res_echantillonnage, c, d, max = extraction_1freq(
                patchs_size, phase, zero, filtering, echogram, echograms, depths, athwart, along, masks, freq)
        else:
            CleanBottom, Bottom, Depth, Echogram, Mask_Cleaning, res_echantillonnage, c, d, max = extraction_1freq(
                patchs_size, phase, zero, filtering, echogram, echograms, depths, athwart, along, masks, freq)
        Mat_clean_j = filtre_bottom_1echogram(Depth, Echogram, CleanBottom, res_echantillonnage, c, d)
        if np.all(Mat_clean_j == 0):
            no_bottom += 1
        labels.append(np.array(Mat_clean_j))
        train.append(np.array(Echogram.T))
        if Echogram.shape != (patchs_size, patchs_size):
            print(f'size : {c, d}, {np.max(Depth)/res_echantillonnage}')
            print(f'min, max bottom : {int(np.nanmin(Bottom) / res_echantillonnage), int(np.nanmax(Bottom) / res_echantillonnage)}')
            print(f'med : {(int(np.nanmedian(Bottom) / res_echantillonnage) - (patchs_size // 2)), (int(np.nanmedian(Bottom) / res_echantillonnage) - (patchs_size // 2)) + patchs_size}')
            break
        nb_imgs += 1
        zero += patchs_size
        print("Nombre d'images générées : "+str(nb_imgs))
    print(f"Nombre d'images sans fond : {no_bottom}")

    # Enregistrer les nouvelles images comme un fichier TIFF
    with h5py.File(train_name + '.h5', 'w') as h5file:
        h5file.create_dataset('images', data=np.array(train).reshape((nb_imgs, 1, patchs_size, patchs_size)).astype('uint8'), compression="gzip", compression_opts=9, dtype='uint8')
        h5file.create_dataset('labels', data=np.array(labels).reshape((nb_imgs, 1, patchs_size, patchs_size)).astype('uint8'), compression="gzip", compression_opts=9, dtype='uint8')

### Création des training et validation sets
def donnees(patchs_size, phase, creation_fonc, freq, nb_train, nb_validation, zero_ini, filtering, echogram, echograms, depths, athwart, along, masks, train_name, validation_name):
    if not phase and creation_fonc==creation_imgs_fond_train_phase:
        print("Erreur, les données fournies ne contiennent pas l'information de phases")
    else:
        # Train data
        print('------- Training set -------')
        ## Extraction patchs
        creation_fonc(patchs_size, phase, freq, nb_train, zero_ini, filtering, echogram, echograms, depths, athwart, along, masks, train_name)
        with h5py.File(train_name+'.h5', 'r') as h5file:
            dataset = h5file['images']
            dataset_size = dataset.shape
            print(f"Shape : {dataset_size}")
            
        # Validation data
        print('------- Validation set -------')
        if zero_ini > nb_validation:
            zero_validation = 0
        else:
            zero_validation = zero_ini + nb_train + 1
        creation_fonc(patchs_size, phase, freq, nb_validation, zero_validation, filtering, echogram, echograms, depths, athwart, along, masks, validation_name)
        with h5py.File(validation_name+'.h5', 'r') as h5file:
            dataset = h5file['images']
            dataset_size = dataset.shape
            print(f"Shape : {dataset_size}")

if __name__ == '__main__':
    print('\n----------------Lecture données-----------------\n')
    scopes = False
    patchs_size = 100

    train_name = "train-volume"
    label_name = "train-labels"
    validation_name = "validation-volume"
    label_validation_name = 'validation-labels'
    if scopes:
        filtering = 'Filtering_SCOPES.mat'
        echogram = 'Echogram_SCOPES.mat'
        echograms = ['Echogram18', 'Echogram38', 'Echogram70',
                     'Echogram120', 'Echogram200', 'Echogram333']
        depths = ['Depth18', 'Depth38', 'Depth70',
                  'Depth120', 'Depth200', 'Depth333']
        athwart = ['AthAngle18', 'AthAngle38', 'AthAngle70',
                   'AthAngle120', 'AthAngle200', 'AthAngle333']
        along = ['AlAngle18', 'AlAngle38', 'AlAngle70',
                 'AlAngle120', 'AlAngle200', 'AlAngle333']
        masks = ['Mask_Cleaning18', 'Mask_Cleaning38', 'Mask_Cleaning70',
                 'Mask_Cleaning120', 'Mask_Cleaning200', 'Mask_Cleaning333']
        phase = False
        nb_freq = 6

        freq = 4
        pourcentage = 0.7
        ping_ini, ping_fin = 330464, 1986151
        nb_ping = ping_fin - ping_ini
        nb_train, zero_ini, nb_validation = int(pourcentage * (nb_ping // patchs_size)), ping_ini, int(
            (1 - pourcentage) * (nb_ping // patchs_size)) - 1
        donnees(patchs_size, phase, creation_imgs_fond_train_1_freq, freq, nb_train, nb_validation, zero_ini, filtering, echogram, echograms, depths, athwart, along, masks, train_name, validation_name)

    else :
        filtering = 'Filtering_FAROFA3.mat'
        echogram = 'Echogram_FAROFA3.mat'
        echograms = ['Echogram70', 'Echogram200']
        depths = ['Depth70', 'Depth200']
        athwart = ['AthAngle70', 'AthAngle200']
        along = ['AlAngle70', 'AlAngle200']
        masks = ['Mask_Cleaning70', 'Mask_Cleaning200']
        phase = False
        nb_freq = 2

        pourcentage = 0.7
        freq = 1
        nb_ping = nb_ping_modif(filtering, echogram, echograms)
        print("Nombre de ping de la campagne : "+str(nb_ping))
        nb_train, zero_ini, nb_validation = int(pourcentage * (nb_ping // patchs_size)), 0, int((1 - pourcentage) * (nb_ping // patchs_size)) - 1
        zero_validation = zero_ini + nb_train + 1
        donnees(patchs_size, phase, creation_imgs_fond_train_1_freq, freq, nb_train, nb_validation, zero_ini, filtering, echogram, echograms, depths, athwart, along, masks, train_name, validation_name)