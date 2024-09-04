# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % parametrage.py
# % -------------------------------
# % Summary : script that determines the parameters to use to train the UNet CNN with Pytorch
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % INPUTS:
# % - args.epoch : number of epochs during which the CNN is trained
# % - args.rootdir : folder containing the data to train, validate or test the CNN
# % - args.filter_sizes : sizes of the convolutional layers of the CNN
# % OUTPUTS:
# % - losses_train : array numpy containing the training and validation losses of each model trained
# % - durations : array numpy containing the duration of each model trained
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import argparse
import torch
from torch import cuda
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import psutil
import random
import socket
import matplotlib.colors as colors

from unet import UNet
from echograms import Echograms
from metric import iou, pix_acc
from loss import Binary_Cross_Entropy_Loss


def parse_args():
    """Parse command line arguments for training the image segmentation model."""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--test-batch-size', type=int, default=64, metavar='N',
        help='Input batch size for testing (default: 64)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=1,
        help='Number of segmentation classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of workers to load data'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='Disables CUDA training'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='Automatic mixed precision training'
    )
    parser.add_argument(
        '--opt-level', type=str
    )
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='Keep batch norm layers with 32-bit precision'
    )
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='Random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='How many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save', action='store_true', default=True,
        help='Save the current model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='Record training log to Tensorboard'
    )
    args = parser.parse_args()
    return args


def seed_worker(seed=0):
    """Set random seeds for reproducibility and configure PyTorch for deterministic algorithms."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True


def get_trainloader(batch_size, num_worker):
    """Create a DataLoader for the training data.

    Args:
        batch_size (int): Number of samples per batch.
        num_worker (int): Number of workers for loading data.

    Returns:
        DataLoader: DataLoader object for training data.
        int: Total number of samples in the dataset.
    """
    data = Echograms(root_dir='/home/lenais/detection_fond/')
    n = len(data)
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  pin_memory=True,
                                  generator=g
                                  )
    return train_dataloader, n


def colormaps():
    """Create custom colormaps for visualization.

    Returns:
        tuple: Two colormaps with red and green colors with transparency.
    """
    red_colors = plt.cm.Reds(np.linspace(0, 1, 256))
    green_colors = plt.cm.Greens(np.linspace(0, 1, 256))
    for i in range(red_colors.shape[0]):
        alpha = i / (red_colors.shape[0] - 1)
        red_colors[i, 3] = alpha  # Set alpha to create a fade effect
        green_colors[i, 3] = alpha
    red_transparent = colors.LinearSegmentedColormap.from_list('RedTransparent', red_colors)
    green_transparent = colors.LinearSegmentedColormap.from_list('GreenTransparent', green_colors)
    return red_transparent, green_transparent


def train(model, device, dataloader, optimizer, criterion):
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        device (str): Device to train the model ('cpu' or 'cuda').
        dataloader (DataLoader): DataLoader to load training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Loss value after training.
    """
    model.train()
    loss = 0.
    for step, sample in enumerate(dataloader):
        X = sample['image'].to(device)
        y = sample['label'].to(device)
        optimizer.zero_grad()
        y_pred = model.double().to(device)(X)
        loss = criterion(y_pred, y.to(torch.float))

        loss.backward()
        optimizer.step()
    return loss.item()


def initialize_model(args):
    """Initialize the model checkpoint dictionary to store training progress.

    Args:
        args (object): Arguments object containing training configuration.

    Returns:
        dict: Dictionary to store model training progress and parameters.
    """
    model_dict = {
        'total_epoch': args.epoch,
        'n_classes': args.n_classes,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'train_loss': list(),
        'test_loss': list(),
        'epoch_duration': list(),
        'metrics': {
            'IOU': list(),
            'pix_acc': list(),
            'best': {
                'IOU': 0.,
                'pixel_acc': 0.,
                'epoch': 0
            }
        }
    }
    return model_dict


def find_free_port():
    """Find a free port on the current machine.

    Returns:
        int: An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def initialize_weights(model):
    """Initialize the weights of the model using Xavier uniform distribution.

    Args:
        model (torch.nn.Module): Model whose weights are to be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def get_model(args, device, in_channels, world_size):
    """Initialize or load a model checkpoint and set up the model and optimizer.

    Args:
        args (object): Arguments object containing model configuration.
        device (str): Device to train and evaluate the model ('cpu' or 'cuda').
        in_channels (int): Number of input channels to the model.
        world_size (int): Number of devices for distributed training.

    Returns:
        tuple: Contains the model, optimizer, and model checkpoint dictionary.
    """
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model(args)
    n_classes = model_dict['n_classes']

    # Setup model
    model = UNet(n_classes, in_channels, filter_sizes=args.filter_sizes).cuda() if device == 'cuda' else UNet(n_classes,
                                                                                                              in_channels,
                                                                                                              filter_sizes=args.filter_sizes)
    model = model.to(device)
    model.apply(initialize_weights)
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer, model_dict


def get_memory_usage():
    """Get the memory usage of the current process.

    Returns:
        int: Memory usage in bytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def train_model(args, freq, world_size):
    """Train the model with different configurations and save training results.

    Args:
        args (object): Arguments object containing training configuration.
        freq (str): Frequency value for the training configuration.
        world_size (int): Number of devices for distributed training.
    """
    # Initialize model
    device = (torch.device(f'cuda') if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    in_channels = 1
    model, optimizer, model_dict = get_model(args, device, in_channels, world_size)
    # Define loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))  # Modification for positive weight

    print(f'{device} : {freq}kHz')
    losses_train = []
    durations = []
    for lr in np.arange(-7, 0, 1):
        model, optimizer, model_dict = get_model(args, device, in_channels, 10. ** lr)
        for batch_size in np.arange(2, 7, 1):
            print(f'Lr : {10. ** lr}, Bs : {int(2 ** batch_size)}')
            # DataLoader
            trainloader, n_train = get_trainloader(int(2 ** batch_size), args.num_workers)
            # Train and evaluate model
            start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
            n_epoch = start_epoch + args.epoch - 1
            t_ini = time.time()
            losses_1model = []
            for epoch in range(start_epoch, n_epoch + 1):
                train_loss = train(model, device, trainloader, optimizer, criterion)
                losses_1model.append(train_loss)
                print(len(losses_1model))
            t_fin = time.time()
            durations.append(t_fin - t_ini)
            losses_train.append(losses_1model)
    np.array(losses_train)
    np.save('/home/lenais/detection_fond/param', losses_train, allow_pickle=True)
    np.save('/home/lenais/detection_fond/durations', durations, allow_pickle=True)


if __name__ == '__main__':
    args = parse_args()
    args.lr = 0.0001
    args.batch_size = 32
    args.epoch = 10
    args.rootdir = '/home/lenais/detection_fond/'
    args.filter_sizes = [8, 16, 32, 64, 128]
    args.pos_weight = 1.

    world_size = torch.cuda.device_count()
    freq = '200'
    os.environ['MASTER_ADDR'] = 'glcblade14'
    os.environ['MASTER_PORT'] = f'{find_free_port()}'
    train_model(args=args, freq=freq, world_size=world_size)

    # Display of settings calculation
    losses_train = np.load('/home/lenais/detection_fond/param.npy')
    print(losses_train.shape)
    epoques = np.arange(start=1, stop=args.epoch + 1, step=1)
    durations = np.load('/home/lenais/detection_fond/durations.npy')
    print(durations.shape)

    i = 0
    for lr in np.arange(-7, 0, 1):
        plt.figure()
        for batch_size in np.arange(2, 7, 1):
            plt.plot(epoques, losses_train[i], label='Batch_size = ' + str(2 ** batch_size))
            i += 1
        plt.xticks(epoques)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Training losses  lr = " + str(10. ** lr))

    plt.figure()
    plt.plot(durations)
    plt.ylabel('Duration of training (s)')
    plt.title("Training durations")

    plt.show()
    print('Done')
