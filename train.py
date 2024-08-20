import os
import argparse
import torch
from torch import cuda
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib
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
matplotlib.use('Agg')

def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--test-batch-size', type=int, default=64, metavar='N',
        help='input batch size for testing (default: 3)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=1,
        help='number of segmentation classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of workers to load data'
    )
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='automatic mixed precision training'
    )
    parser.add_argument(
        '--opt-level', type=str
    )
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='keep batch norm layers with 32-bit precision'
    )
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )
    parser.add_argument(
        '--save', action='store_true', default=True,
        help='save the current model'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='record training log to Tensorboard'
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

def get_trainloader(batch_size, num_worker):
    data = Echograms(root_dir='D:/PFE/Codes_finaux/data/')
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

def get_validationloader(batch_size, num_worker):
    data = Echograms(data_type='validate', root_dir='D:/PFE/Codes_finaux/data/')
    n = len(data)
    g = torch.Generator()
    g.manual_seed(0)
    validation_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  pin_memory=True,
                                  generator=g
                                  )
    return validation_dataloader, n

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

def train(model, device, dataloader, optimizer, criterion, epoch, n):
    """train model for one epoch
    Args:
        model (torch.nn.Module): model to train
        device (str): device to train model ('cpu' or 'cuda')
        data_loader (object): iterator to load data
        optimizer (torch.nn.optim): stochastic optimzation strategy
        criterion (torch.nn.Module): loss function
        epoch (int): current epoch
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

        # Affichage
        log_interval = 1
        if step % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (step+1) * len(X), n,
                100. * (step+1)* len(X) / n, loss.item()))

    return loss.item()

def validate(model, device, dataloader, criterion, n_classes, n):
    """Evaluate model performance with validation data
    Args:
        model (torch.nn.Module): model to evaluate
        device (str): device to evaluate model ('cpu' or 'cuda')
        data_loader (object): iterator to load data
        criterion (torch.nn.Module): loss function
        n_classes (int): number of segmentation classes
    """
    model.eval()
    test_loss = 0
    class_iou = [0.] * n_classes
    pixel_acc = 0.
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            X = sample['image'].to(device)
            y = sample['label'].to(device)
            y_pred = model.to(device)(X)
            test_loss += criterion(y_pred, y.to(torch.float)).item()
            pred = torch.sigmoid(y_pred) > 0.5
            pred = pred.view(X.shape[0], -1)
            y = y.view(X.shape[0], -1)
            batch_iou = iou(pred, y, X.shape[0], n_classes)
            class_iou += batch_iou * (X.shape[0] / n)
            pixel_acc += pix_acc(pred, y, X.shape[0]) * (X.shape[0] / n)

    test_loss /= n
    avg_iou = np.mean(class_iou)

    print('\nValidation set: Average loss: {:.4f}, '.format(test_loss)
        + 'Average IOU score: {:.2f}, '.format(avg_iou)
        + 'Average pixel accuracy: {:.2f}\n'.format(pixel_acc))

    return test_loss, avg_iou, pixel_acc

def initialize_model(args):
    """Initialize model checkpoint dictionary for storing training progress
    Args:
        args (object):
            epoch (int): total number of epochs to train model
            n_classes (int): number of segmentation classes
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def get_model(args, device, in_channels, world_size):
    """Intialize or load model checkpoint and intialize model and optimizer
    Args:
        args (object):
            model (str): filename of model to load
                (initialize new model if none is given)
        device (str): device to train and evaluate model ('cpu' or 'cuda')
    """
    if args.model:
        # Load model checkpoint
        model_path = os.path.join(os.getcwd(), f'models/{args.model}')
        model_dict = torch.load(model_path)
    else:
        model_dict = initialize_model(args)
    n_classes = model_dict['n_classes']

    # Setup model
    model = UNet(n_classes, in_channels, filter_sizes=args.filter_sizes).cuda() if device == 'cuda' else UNet(n_classes, in_channels, filter_sizes=args.filter_sizes)
    model = model.to(device)
    model.apply(initialize_weights)
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer, model_dict

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def train_model(args, freq, world_size):
    # initialize model
    device = (torch.device(f'cuda') if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    in_channels = 1
    model, optimizer, model_dict = get_model(args, device, in_channels, world_size)
    # define loss function
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    # dataloader
    trainloader, n_train = get_trainloader(args.batch_size, args.num_workers)
    validationloader, n_validation = get_validationloader(args.batch_size, args.num_workers)
    print(f'{device} : {freq}kHz')

    # train and evaluate model
    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    n_epoch = start_epoch + args.epoch - 1
    model_path = os.getcwd() + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    t_tot = 0
    test_loss_prev = 1000
    for epoch in range(start_epoch, n_epoch + 1):
        print(f'Train Epoch : {epoch}/{n_epoch}')
        t_ini = time.time()
        train_loss = train(model, device, trainloader, optimizer, criterion, epoch, n_train)
        test_loss, test_iou, test_pix_acc = validate(model, device, validationloader, criterion, args.n_classes, n_validation)
        model_dict['train_loss'].append(train_loss)
        model_dict['test_loss'].append(test_loss)
        model_dict['metrics']['IOU'].append(test_iou)
        model_dict['metrics']['pix_acc'].append(test_pix_acc)
        if epoch == 1 or test_iou > model_dict['metrics']['best']['IOU']:
            model_dict['model_state_dict'] = model.state_dict()
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['metrics']['best']['IOU'] = test_iou
            model_dict['metrics']['best']['pix_acc'] = test_pix_acc
            model_dict['metrics']['best']['epoch'] = epoch
        if args.save and test_loss < test_loss_prev and epoch % 5 == 0:
            model_name = f'models/models_mixte_FAROFA_SCOPES/{model.name}{epoch}_{freq}.pt'
            torch.save(model_dict, model_name)
        t_fin = time.time()
        model_dict['epoch_duration'] = t_fin-t_ini
        t_tot += t_fin - t_ini

    print('Best IOU:', model_dict['metrics']['best']['IOU'])
    print('Pixel accuracy:', model_dict['metrics']['best']['pix_acc'])
    print(f'Complete training duration : {t_tot} s')
    print(f"Utilisation de la mémoire à la fin du script : {get_memory_usage() / (1024 ** 2):.2f} MB")

if __name__ == '__main__':
    args = parse_args()
    args.lr = 0.0001
    args.batch_size = 32
    args.epoch = 25
    args.rootdir = 'D:/PFE/Codes_finaux/'
    args.filter_sizes = [8, 16, 32, 64, 128]
    args.pos_weight = 1.

    world_size = torch.cuda.device_count()
    freq = '200'
    os.environ['MASTER_ADDR'] = 'glcblade14'
    os.environ['MASTER_PORT'] = f'{find_free_port()}'
    train_model(args=args, freq=freq, world_size=world_size)