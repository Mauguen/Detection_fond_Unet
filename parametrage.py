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

# from unet import UNet
# from echograms import Echograms
# from metric import iou, pix_acc
# from loss import Binary_Cross_Entropy_Loss


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)'
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
        '--model', type=str, default=False,
        help='model to retrain'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='record training log to Tensorboard'
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

def get_trainloader(batch_size, num_worker):
    data = Echograms(data_type='param')
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
    }
    return model_dict

def get_model(args, device, in_channels, lr):
    """Intialize or load model checkpoint and intialize model and optimizer

    Args:
        args (object):
            model (str): filename of model to load
                (initialize new model if none is given)
        device (str): device to train and evaluate model ('cpu' or 'cuda')
    """
    model_dict = initialize_model(args)
    n_classes = model_dict['n_classes']

    model = UNet(n_classes, in_channels).cuda() if device == 'cuda' else UNet(
        n_classes, in_channels)
    optimizer = optim.Adam(model.parameters(), lr)
    return model, optimizer, model_dict

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

if __name__ == '__main__':
    args = parse_args()
    # if args.tensorboard:
    #     writer = SummaryWriter()
    # # initialize model
    # device = ('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # print(device)
    # in_channels = 1
    # # define loss function
    # criterion = Binary_Cross_Entropy_Loss()
    # lr =0.000001
    # losses_train = []
    # lrs = []
    # durations = []
    # while lr < 0.0001:
    #     model, optimizer, model_dict = get_model(args, device, in_channels, lr)
    #     for batch_size in np.arange(5, 65, 5):
    #         # dataloader
    #         trainloader, n_train = get_trainloader(int(batch_size), args.num_workers)
    #         # train and evaluate model
    #         start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1
    #         n_epoch = start_epoch + args.epochs - 1
    #         t_ini = time.time()
    #         losses_1model = []
    #         for epoch in range(start_epoch, n_epoch + 1):
    #             train_loss = train(model, device, trainloader, optimizer, criterion, epoch, n_train)
    #             losses_1model.append(train_loss)
    #             print(len(losses_1model))
    #         t_fin = time.time()
    #         durations.append(t_fin-t_ini)
    #         losses_train.append(losses_1model)
    #     lrs.append(lr)
    #     lr = lr*10
    # np.array(losses_train)
    # np.save('/home/lenais/detection_fond/param', losses_train, allow_pickle=True)
    # np.save('/home/lenais/detection_fond/durations', durations, allow_pickle=True)

    losses_train = np.load('E:/PFE/Codes_backup/Extraction_fond/UNet-_final/param.npy')
    print(losses_train.shape)
    durations = np.load('E:/PFE/Codes_backup/Extraction_fond/UNet-_final/durations.npy')
    print(durations.shape)
    epoques = np.arange(start=1, stop=args.epochs + 1, step=1)
    # print(lrs)
    lrs = np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01])

    i = 0
    fig, axs = plt.subplots(2, 3, sharex=True)
    plt.tight_layout()
    for j, lr in enumerate(lrs):
        for batch_size in np.arange(5, 65, 5):
            if batch_size != 5 :
                axs[j//3, j%3].plot(epoques, losses_train[i], label='Batch_size = '+str(batch_size))
            i+=1
        axs[j//3, j%3].set_title("Training losses  lr = " + str(lr))
        if j%3 == 0:
            axs[j//3, j%3].set_ylabel('Loss')
        if j >= 3:
            plt.xticks(epoques)
            axs[j//3, j%3].set_xlabel('Epochs')
        if lr == 0.01:
            plt.legend()

    plt.figure()
    plt.plot(np.arange(10, 65, 5), durations[:len(np.arange(10, 65, 5))])
    plt.ylabel('Duration of training (s)')
    plt.xlabel('Batch size')
    plt.title("Training durations")

    # print(np.sum(durations))

    plt.show()
    print('Done')
