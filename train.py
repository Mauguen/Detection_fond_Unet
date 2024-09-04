# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % train.py
# % -------------------------------
# % Summary : script used to train a UNet CNN with Pytorch
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % INPUTS:
# % - args.lr : learning rate of the CNN
# % - args.batch_size : batch-size of the CNN
# % - args.epoch : number of epochs during which the CNN is trained
# % - args.rootdir : folder containing the data to train, validate or test the CNN
# % - args.filter_sizes : sizes of the convolutional layers of the CNN
# % OUTPUTS:
# % - model
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

matplotlib.use('Agg')  # Use a non-interactive backend for Matplotlib (useful for environments without a GUI)


# Function to parse command line arguments
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train image segmentation')

    # Argument for test batch size
    parser.add_argument(
        '--test-batch-size', type=int, default=64, metavar='N',
        help='input batch size for testing (default: 64)'
    )

    # Argument for SGD momentum
    parser.add_argument(
        '--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)'
    )

    # Argument for the number of segmentation classes
    parser.add_argument(
        '--n-classes', type=int, default=1,
        help='number of segmentation classes'
    )

    # Argument for the number of workers for data loading
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='number of workers to load data'
    )

    # Argument to disable CUDA (GPU) usage
    parser.add_argument(
        '--no-cuda', action='store_true', default=False,
        help='disables CUDA training'
    )

    # Argument for automatic mixed precision (AMP) training
    parser.add_argument(
        '--amp', action='store_true', default=False,
        help='automatic mixed precision training'
    )

    # Argument to set AMP optimization level
    parser.add_argument(
        '--opt-level', type=str
    )

    # Argument to keep batch normalization layers in 32-bit precision
    parser.add_argument(
        '--keep_batchnorm_fp32', type=str, default=None,
        help='keep batch norm layers with 32-bit precision'
    )

    # Argument to set the loss scale for AMP
    parser.add_argument(
        '--loss-scale', type=str, default=None
    )

    # Argument to set the random seed
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)'
    )

    # Argument to set the logging interval during training
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status'
    )

    # Argument to enable saving the current model
    parser.add_argument(
        '--save', action='store_true', default=True,
        help='save the current model'
    )

    # Argument to specify a model for retraining
    parser.add_argument(
        '--model', type=str, default=None,
        help='model to retrain'
    )

    # Argument to log training status to Tensorboard
    parser.add_argument(
        '--tensorboard', action='store_true', default=True,
        help='record training log to Tensorboard'
    )

    args = parser.parse_args()
    return args


# Function to set a random seed to ensure reproducibility
def seed_worker(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For deterministic behavior with CuDNN
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # Ensure deterministic algorithms
    torch.utils.deterministic.fill_uninitialized_memory = True


# Function to get the DataLoader for training
def get_trainloader(batch_size, num_worker):
    data = Echograms(root_dir='D:/PFE/Codes_finaux/data/')  # Load training data
    n = len(data)  # Total number of samples in the training set
    g = torch.Generator()  # Generator to ensure reproducibility
    g.manual_seed(0)
    train_dataloader = DataLoader(data,
                                  batch_size=batch_size,
                                  shuffle=True,  # Shuffle data at every epoch
                                  num_workers=num_worker,
                                  worker_init_fn=seed_worker,  # Use the seed for workers
                                  pin_memory=True,
                                  generator=g
                                  )
    return train_dataloader, n


# Function to get the DataLoader for validation
def get_validationloader(batch_size, num_worker):
    data = Echograms(data_type='validate', root_dir='D:/PFE/Codes_finaux/data/')  # Load validation data
    n = len(data)  # Total number of samples in the validation set
    g = torch.Generator()
    g.manual_seed(0)
    validation_dataloader = DataLoader(data,
                                       batch_size=batch_size,
                                       shuffle=False,  # No shuffling for validation
                                       num_workers=num_worker,
                                       worker_init_fn=seed_worker,
                                       pin_memory=True,
                                       generator=g
                                       )
    return validation_dataloader, n


# Function to generate transparent colormaps for masks
def colormaps():
    red_colors = plt.cm.Reds(np.linspace(0, 1, 256))
    green_colors = plt.cm.Greens(np.linspace(0, 1, 256))
    for i in range(red_colors.shape[0]):
        alpha = i / (red_colors.shape[0] - 1)  # Gradation of transparency
        red_colors[i, 3] = alpha
        green_colors[i, 3] = alpha
    red_transparent = colors.LinearSegmentedColormap.from_list('RedTransparent', red_colors)
    green_transparent = colors.LinearSegmentedColormap.from_list('GreenTransparent', green_colors)
    return red_transparent, green_transparent


# Function to train the model for one epoch
def train(model, device, dataloader, optimizer, criterion, epoch, n):
    """Train model for one epoch"""
    model.train()  # Set the model to training mode
    loss = 0.
    for step, sample in enumerate(dataloader):
        X = sample['image'].to(device)  # Get images and move them to the device (GPU/CPU)
        y = sample['label'].to(device)  # Get labels and move them to the device
        optimizer.zero_grad()  # Reset gradients
        y_pred = model.double().to(device)(X)  # Make predictions with the model
        loss = criterion(y_pred, y.to(torch.float))  # Calculate the loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights

        # Periodically log the loss
        log_interval = 1
        if step % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (step + 1) * len(X), n,
                       100. * (step + 1) * len(X) / n, loss.item()))

    return loss.item()


# Function to validate the model
def validate(model, device, dataloader, criterion, n_classes, n):
    """Evaluate model performance with validation data"""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    class_iou = [0.] * n_classes
    pixel_acc = 0.
    with torch.no_grad():  # Disable gradient calculation
        for step, sample in enumerate(dataloader):
            X = sample['image'].to(device)  # Get images and move them to the device
            y = sample['label'].to(device)  # Get labels and move them to the device
            y_pred = model.to(device)(X)  # Make predictions with the model
            test_loss += criterion(y_pred, y.to(torch.float)).item()  # Calculate the loss
            pred = torch.sigmoid(y_pred) > 0.5  # Apply sigmoid function and threshold to get predictions
            pred = pred.view(X.shape[0], -1)
            y = y.view(X.shape[0], -1)
            batch_iou = iou(pred, y, X.shape[0], n_classes)  # Calculate IoU for the batch
            class_iou += batch_iou * (X.shape[0] / n)
            pixel_acc += pix_acc(pred, y, X.shape[0]) * (X.shape[0] / n)

    test_loss /= n
    avg_iou = np.mean(class_iou)

    print('\nValidation set: Average loss: {:.4f}, '.format(test_loss))
    print('Class IoU:', class_iou)
    print('Pixel Accuracy:', pixel_acc)

    return test_loss, avg_iou, pixel_acc


# Function to initialize model weights using Xavier initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Function to initialize model parameters
def initialize_model(args):
    model_dict = {}
    model_dict['best_prec1'] = 0.0  # Initialize best precision to 0
    model_dict['best_prec2'] = 0.0  # Initialize best precision to 0
    model_dict['start_epoch'] = 0  # Start at epoch 0
    model_dict['train_losses'] = []  # Initialize training losses list
    model_dict['validation_losses'] = []  # Initialize validation losses list
    model_dict['validation_acc'] = []  # Initialize validation accuracy list
    model_dict['best_loss'] = 100  # Initialize best loss to a high value
    return model_dict


# Function to retrieve or initialize a U-Net model
def get_model(args, device, in_channels, world_size):
    model = UNet(in_channels=in_channels, n_classes=args.n_classes).to(device)  # Create a U-Net model
    model = torch.compile(model, backend="aot_eager", mode="reduce-overhead").to(
        device)  # Compile the model for optimized performance
    optimizer = torch.optim.Adam(model.parameters())  # Use Adam as the optimizer

    if args.model is not None:
        if os.path.isfile(args.model):
            print("=> loading model '{}'".format(args.model))
            model_dict = torch.load(args.model)
            model.load_state_dict(model_dict['state_dict'])  # Load pretrained model weights
            optimizer.load_state_dict(model_dict['optimizer'])
            print("=> loaded model '{}' (epoch {})"
                  .format(args.model, model_dict['start_epoch']))
        else:
            print("=> no model found at '{}'".format(args.model))
            model_dict = initialize_model(args)  # Initialize model parameters
            initialize_weights(model)  # Initialize model weights
    else:
        model_dict = initialize_model(args)  # Initialize model parameters
        initialize_weights(model)  # Initialize model weights

    return model, model_dict, optimizer

# Function to get the current memory usage of the running process
def get_memory_usage():
    process = psutil.Process(os.getpid())  # Get the current process
    mem_info = process.memory_info()  # Retrieve memory usage information for the process
    return mem_info.rss  # Return the Resident Set Size (RSS) in bytes, which is the non-swapped physical memory used by the process


# Main function to train the model
def train_model(args, freq, world_size):
    # Initialize the model
    # Select the device to use (GPU if available and not disabled, otherwise CPU)
    device = (torch.device(f'cuda') if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    in_channels = 1  # Define the number of input channels (1 for grayscale images)

    # Call the get_model function to retrieve the model, optimizer, and model parameters
    model, optimizer, model_dict = get_model(args, device, in_channels, world_size)

    # Define the loss function
    # Use Binary Cross Entropy with logits, considering a weight for the positive class
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))

    # DataLoader for the training and validation datasets
    trainloader, n_train = get_trainloader(args.batch_size, args.num_workers)  # Load the training data
    validationloader, n_validation = get_validationloader(args.batch_size, args.num_workers)  # Load the validation data

    # Print the device and frequency being used
    print(f'{device} : {freq}kHz')

    # Train and evaluate the model
    start_epoch = 1 if not args.model else model_dict['total_epoch'] + 1  # Determine the starting epoch
    n_epoch = start_epoch + args.epoch - 1  # Calculate the total number of epochs to run

    # Set up the directory to save models
    model_path = os.getcwd() + '/models'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)  # Create the directory if it doesn't exist

    t_tot = 0  # Initialize total training time
    test_loss_prev = 1000  # Initialize a high previous test loss for comparison

    # Loop over each epoch
    for epoch in range(start_epoch, n_epoch + 1):
        print(f'Train Epoch : {epoch}/{n_epoch}')  # Print the current epoch
        t_ini = time.time()  # Record the start time for the epoch

        # Train the model for one epoch and get the training loss
        train_loss = train(model, device, trainloader, optimizer, criterion, epoch, n_train)

        # Validate the model and get the validation loss, IOU, and pixel accuracy
        test_loss, test_iou, test_pix_acc = validate(model, device, validationloader, criterion, args.n_classes,
                                                     n_validation)

        # Store the results in the model dictionary
        model_dict['train_loss'].append(train_loss)
        model_dict['test_loss'].append(test_loss)
        model_dict['metrics']['IOU'].append(test_iou)
        model_dict['metrics']['pix_acc'].append(test_pix_acc)

        # Save the model if it's the first epoch or if it has the best IOU so far
        if epoch == 1 or test_iou > model_dict['metrics']['best']['IOU']:
            model_dict['model_state_dict'] = model.state_dict()
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['metrics']['best']['IOU'] = test_iou
            model_dict['metrics']['best']['pix_acc'] = test_pix_acc
            model_dict['metrics']['best']['epoch'] = epoch

        # Save the model if certain conditions are met (e.g., every 5 epochs and improvement in test loss)
        if args.save and test_loss < test_loss_prev and epoch % 5 == 0:
            model_name = f'models/{model.name}{epoch}_{freq}.pt'
            torch.save(model_dict, model_name)

        t_fin = time.time()  # Record the end time for the epoch
        model_dict['epoch_duration'] = t_fin - t_ini  # Calculate the epoch duration
        t_tot += t_fin - t_ini  # Update the total training time

    # Print the best IOU and pixel accuracy achieved during training
    print('Best IOU:', model_dict['metrics']['best']['IOU'])
    print('Pixel accuracy:', model_dict['metrics']['best']['pix_acc'])
    print(f'Complete training duration : {t_tot} s')  # Print the total training duration

    # Print the memory usage at the end of the script
    print(f"Memory usage at the end of the script: {get_memory_usage() / (1024 ** 2):.2f} MB")


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    args.lr = 0.0001  # Set learning rate
    args.batch_size = 32  # Set batch size
    args.epoch = 25  # Set the number of epochs
    args.rootdir = 'D:/PFE/Codes_finaux/'  # Set the root directory for the data
    args.filter_sizes = [8, 16, 32, 64, 128]  # Define filter sizes for the model
    args.pos_weight = 1.  # Set the weight for the positive class in the loss function

    world_size = torch.cuda.device_count()  # Get the number of GPUs available
    freq = '200'  # Set the frequency (in kHz)
    os.environ['MASTER_ADDR'] = 'glcblade14'  # Set the master address for distributed training
    os.environ['MASTER_PORT'] = f'{find_free_port()}'  # Set the master port

    # Start training the model
    train_model(args=args, freq=freq, world_size=world_size)