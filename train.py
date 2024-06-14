import os
import argparse

import torch
from torch import nn, optim, DoubleTensor
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import random

from unet import UNet
from echograms import Echograms
from metric import iou, pix_acc
from loss import Binary_Cross_Entropy_Loss


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--batch-size', type=int, default=10, metavar='N',
        help='input batch size for training (default: 3)'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=10, metavar='N',
        help='input batch size for testing (default: 3)'
    )
    parser.add_argument(
        '--epochs', type=int, default=1, metavar='N',
        help='number of epochs to train (default: 10)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0001, metavar='LR',
        help='learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )
    parser.add_argument(
        '--n-classes', type=int, default=2,
        help='number of segmentation classes'
    )
    parser.add_argument(
        '--num_workers', type=int, default=6,
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
        '--model', type=str, default='UNet7.pt',
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='record training log to Tensorboard'
    )
    args = parser.parse_args()
    return args

def seed_worker(worker_id):
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

def get_trainloader(batch_size, num_worker):
    data = Echograms()
    n = len(data)
    g = torch.Generator()
    g.manual_seed(0)
    train_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                  )
    return train_dataloader, n

def get_validationloader(batch_size, num_worker):
    data = Echograms(data_type='validate')
    n = len(data)
    g = torch.Generator()
    g.manual_seed(0)
    validation_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,
                                  generator=g
                                  )
    return validation_dataloader, n

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
        y = sample['label'].to(device).squeeze(1).long()  # remove channel dimension
        y_pred = model.double()(X)

        # back propogation
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Affichage
        # log_interval = 1
        # if step % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (step+1) * len(X), n,
                100. * (step+1)* len(X) / n, loss.item()))
        step+=1

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
            y = sample['label'].to(device).squeeze(1).long()  # remove channel dimension
            y_pred = model.double()(X)
            test_loss += criterion(y_pred, y).item()  # sum up batch loss
            pred = torch.argmax(y_pred, dim=1)
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
        'total_epoch': args.epochs,
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


def get_model(args, device, in_channels):
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
    # model_dict = initialize_model(args)
    n_classes = model_dict['n_classes']

    model = UNet(n_classes, in_channels).cuda() if device == 'cuda' else UNet(
        n_classes, in_channels)
    optimizer = optim.Adam(model.parameters(), args.lr)
    return model, optimizer, model_dict

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

if __name__ == '__main__':
    # args = parse_args()
    # if args.tensorboard:
    #     writer = SummaryWriter()
    # # Prevent nondeterministic algorithms
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # # initialize model
    # device = ("cuda:0" if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # print(device)
    # in_channels = 1
    # model, optimizer, model_dict = get_model(args, device, in_channels)
    # # define loss function
    # criterion = Binary_Cross_Entropy_Loss()
    # # dataloader
    # trainloader, n_train = get_trainloader(args.batch_size, args.num_workers)
    # validationloader, n_validation = get_validationloader(args.batch_size, args.num_workers)
    # # train and evaluate model
    # start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    # n_epoch = start_epoch + args.epochs - 1
    # model_path = os.getcwd() + '/models'
    # if not os.path.isdir(model_path):
    #     os.mkdir(model_path)
    # model_name = f'models/{model.name}{n_epoch}.pt'
    # t_tot = 0
    # for epoch in range(start_epoch, n_epoch + 1):
    #     t_ini = time.time()
    #     train_loss = train(model, device, trainloader, optimizer, criterion, epoch, n_train)
    #     test_loss, test_iou, test_pix_acc = validate(model, device, validationloader, criterion, args.n_classes, n_validation)
    #     # update tensorboard
    #     if args.tensorboard:
    #         writer.add_scalar('Loss/train', train_loss, epoch)
    #         writer.add_scalar('Loss/test', test_loss, epoch)
    #         writer.add_scalar('IOU/test', test_iou, epoch)
    #         writer.add_scalar('Pixel_Accuracy/test', test_pix_acc, epoch)
    #     # record training progress
    #     model_dict['train_loss'].append(train_loss)
    #     model_dict['test_loss'].append(test_loss)
    #     model_dict['metrics']['IOU'].append(test_iou)
    #     model_dict['metrics']['pix_acc'].append(test_pix_acc)
    #     if epoch == 1 or test_iou > model_dict['metrics']['best']['IOU']:
    #         model_dict['model_state_dict'] = model.state_dict()
    #         model_dict['optimizer_state_dict'] = optimizer.state_dict()
    #         model_dict['metrics']['best']['IOU'] = test_iou
    #         model_dict['metrics']['best']['pix_acc'] = test_pix_acc
    #         model_dict['metrics']['best']['epoch'] = epoch
    #     if args.save:
    #         torch.save(model_dict, model_name)
    #     t_fin = time.time()
    #     model_dict['epoch_duration'] = t_fin-t_ini
    #     print('Epoch duration', t_fin - t_ini, '\n')
    #     t_tot += t_fin - t_ini
    # if args.tensorboard:
    #     writer.close()
    # print('Best IOU:', model_dict['metrics']['best']['IOU'])
    # print('Pixel accuracy:', model_dict['metrics']['best']['pix_acc'])
    # print('Complete training duration : ', t_tot)
    # print(f"Utilisation de la mémoire à la fin du script : {get_memory_usage() / (1024 ** 2):.2f} MB")

    model_dict = torch.load('models/UNet7.pt', map_location=torch.device('cpu'))
    plt.figure()
    epoques = np.arange(start=1, stop=8, step=1)
    plt.plot(epoques, model_dict['train_loss'], label='Train loss')
    plt.plot(epoques, model_dict['test_loss'], label='Validation loss')
    plt.xticks(epoques)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title("Training and validation losses ")
    plt.show()
    print('Done')
