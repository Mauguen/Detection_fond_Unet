# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % unet.py
# % -------------------------------
# % Summary : classe UNet used to create the architecture of the CNN
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % INPUTS:
# % - n_classes : number of classes to distinguish in the data
# % - in_channels : number of channels in one sample (if monofrequency images : in_channels = 1 channel, if multifrequency : in_channels = number of frequencies)
# % OUTPUTS:
# % - model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
import numpy as np


class UNet(torch.nn.Module):
    """Implementation of the U-Net architecture
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
    https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, n_classes, in_channels, batch_norm=False, filter_sizes=None):
        """
        Initializes the U-Net model.

        Args:
            n_classes (int): Number of output classes for segmentation.
            in_channels (int): Number of input channels (e.g., 1 for grayscale images).
            batch_norm (bool): Whether to use batch normalization.
            filter_sizes (list of int): List of filter sizes for each block. If None, a default list is used.
        """
        self.name = 'UNet'
        self.n_classes = n_classes
        self.in_channels = in_channels

        # Set default filter sizes if none are provided
        if filter_sizes is None:
            self.filter_sizes = [8, 16, 32, 64, 128]
        else:
            self.filter_sizes = filter_sizes

        self.n_block = len(self.filter_sizes)  # Number of blocks in the U-Net
        self.batch_norm = batch_norm

        super(UNet, self).__init__()

        # Define the contraction (downsampling) and expansion (upsampling) paths
        self.contract_blocks = self.contract()
        self.expand_blocks = self.expand()

        # Final segmentation layer (1x1 convolution)
        self.segment = nn.Conv2d(
            self.filter_sizes[0],
            self.n_classes,
            kernel_size=1
        )

    def forward(self, x):
        """Performs a forward pass through the network.
        Args:
            x (Tensor): Input image tensor.
        Returns:
            Tensor: Segmented output.
        """
        xs = []  # To store intermediate outputs for skip connections

        # Contracting path (downsampling)
        for block in self.contract_blocks:
            new_x = block(x)
            xs.append(new_x)  # Save output for skip connections
            x = new_x

        # Expanding path (upsampling)
        for i, block in enumerate(self.expand_blocks):
            x = block['up'](x)  # Upsample
            k = self.n_block - i - 2
            x = self.concat(xs[k], x)  # Concatenate with the corresponding feature map from the contracting path
            x = block['conv'](x)  # Apply convolutions

        # Final segmentation
        y_pred = self.segment(x)

        return y_pred

    def concat(self, x, y):
        """Crop and concatenate two feature maps (skip connection).
        Args:
            x (Tensor): Feature map from contracting path.
            y (Tensor): Feature map from expanding path.
        Returns:
            Tensor: Concatenated feature map.
        """
        # Calculate difference in dimensions and pad if necessary
        diffy = x.size()[2] - y.size()[2]
        diffx = x.size()[3] - y.size()[3]
        y = nn.functional.pad(y, (diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2))

        # Concatenate along the channel dimension
        return torch.cat([x, y], dim=1)

    def contract(self):
        """Define the contraction (downsampling) blocks in U-Net.
        Returns:
            list: List of sequential blocks for the contracting path.
        """
        blocks = []
        old = self.in_channels  # Start with the input channels

        # Create each contracting block
        for i, size in enumerate(self.filter_sizes):
            mpool = nn.MaxPool2d(kernel_size=2)  # Max pooling for downsampling
            conv1 = nn.Conv2d(old, size, kernel_size=3, padding='same')
            conv2 = nn.Conv2d(size, size, kernel_size=3, padding='same')
            relu = nn.ReLU(True)
            convs = [mpool, conv1, relu, conv2, relu]

            # Add batch normalization if specified
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [mpool, conv1, b_norm, relu, conv2, b_norm, relu]

            # For the first block, skip the initial MaxPool
            if i == 0:
                convs = convs[1:]

            # Create a sequential block and add it to the list
            block = nn.Sequential(*convs)
            blocks.append(block)

            old = size  # Update the number of input channels for the next block
            self.add_module(f'contract{i + 1}', block)  # Register the block as a module

        return blocks

    def expand(self):
        """Define the expansion (upsampling) blocks in U-Net.
        Returns:
            list: List of dictionaries containing upsampling and convolutional layers.
        """
        blocks = []
        expand_filters = self.filter_sizes[self.n_block - 2::-1]  # Reverse filter sizes for upsampling
        old = self.filter_sizes[-1]  # Start with the last filter size in the contracting path

        # Create each expanding block
        for i, size in enumerate(expand_filters):
            up = nn.ConvTranspose2d(old, size, kernel_size=3, stride=2)  # Transposed convolution for upsampling
            self.add_module(f'up{i + 1}', up)  # Register the upsampling layer

            # Convolutions after upsampling
            conv1 = nn.Conv2d(old, size, kernel_size=3, padding='same')
            conv2 = nn.Conv2d(size, size, kernel_size=3, padding='same')
            relu = nn.ReLU(True)
            convs = [conv1, relu, conv2, relu]

            # Add batch normalization if specified
            if self.batch_norm:
                b_norm = nn.BatchNorm2d(size)
                convs = [conv1, b_norm, relu, conv2, b_norm, relu]

            convs = nn.Sequential(*convs)  # Create a sequential block
            self.add_module(f'deconv{i + 1}', convs)  # Register the block as a module
            blocks.append({'up': up, 'conv': convs})  # Append to the list of blocks

            old = size  # Update the number of input channels for the next block

        return blocks


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    n_channels = 1  # Number of input channels (e.g., 1 for grayscale)
    # Generate a random input tensor with batch size 32 and dimensions 100x100
    im = torch.randn(32, n_channels, 100, 100)

    # Instantiate the U-Net model
    model = UNet(n_classes=1, in_channels=n_channels)
    print(list(model.children()))  # Print the model's layers
    import time
    t = time.time()
    # Perform a forward pass through the model
    x = model(im)
    print(time.time() - t)  # Print the time taken for the forward pass
    print(x.shape, im.shape)  # Print the shape of the output and input tensors
    # Clean up
    del model
    del x