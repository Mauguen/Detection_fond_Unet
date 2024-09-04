# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % metric.py
# % -------------------------------
# % Summary : metrics calculated during the training of the CNN
# % -------------------------------
# % Author : Lénais Mauguen - IRD
# % Date : 2024/08/30
# % -------------------------------
# % INPUTS:
# % - predictions (outputs)
# % - labels (targets)
# % - batch_size
# % OUTPUTS:
# % - pix_acc : pixel accuracy
# % - iou : Intersection over Union score (Jaccard index)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import torch
import numpy as np


def pix_acc(outputs, targets, batch_size):
    """Calculate pixel accuracy.

    Args:
        outputs (torch.nn.Tensor): Prediction outputs from the model.
        targets (torch.nn.Tensor): Ground truth labels.
        batch_size (int): The number of samples in the batch.

    Returns:
        float: The average pixel accuracy over the batch.
    """
    acc = 0.  # Initialize the accuracy counter
    for idx in range(batch_size):
        output = outputs[idx]  # Get the output tensor for the current sample
        target = targets[idx]  # Get the target tensor for the current sample
        correct = torch.sum(torch.eq(output, target).long())  # Count the number of correct pixel matches
        # Calculate the pixel accuracy for this sample and accumulate
        acc += correct / np.prod(np.array(output.shape)) / batch_size
    return acc.item()  # Return the pixel accuracy as a float


def iou(outputs, targets, batch_size, n_classes):
    """Calculate Intersection over Union (IoU) for each class.

    Args:
        outputs (torch.nn.Tensor): Prediction outputs from the model.
        targets (torch.nn.Tensor): Ground truth labels.
        batch_size (int): The number of samples in the batch.
        n_classes (int): The number of segmentation classes.

    Returns:
        np.ndarray: IoU score for each class.
    """
    eps = 1e-6  # Small constant to avoid division by zero
    class_iou = np.zeros(n_classes)  # Array to store IoU scores for each class

    for idx in range(batch_size):
        outputs_cpu = outputs[idx].cpu()  # Move output tensor to CPU
        targets_cpu = targets[idx].cpu()  # Move target tensor to CPU

        for c in range(n_classes):
            # Get the indices of pixels belonging to class 'c' in both outputs and targets
            i_outputs = np.where(outputs_cpu == c)
            i_targets = np.where(targets_cpu == c)
            intersection = np.intersect1d(i_outputs, i_targets).size  # Count pixels in intersection
            union = np.union1d(i_outputs, i_targets).size  # Count pixels in union
            # Compute IoU for this class and accumulate
            class_iou[c] += (intersection + eps) / (union + eps)

    # Average the IoU scores over the batch
    class_iou /= batch_size

    return class_iou  # Return the IoU scores as a numpy array


###############################################################################
# Testing the functions
###############################################################################
if __name__ == '__main__':
    x = torch.randint(high=2, size=(3, 1, 5, 5))  # Generate random predictions
    y = torch.randint(high=2, size=(3, 1, 5, 5))  # Generate random ground truth labels
    print(x[0])  # Print the first sample of predictions
    print(y[0])  # Print the first sample of ground truth labels
    print(torch.sum(torch.eq(x[0], y[0]).long()))  # Print the number of correct pixels in the first sample
    print(sum(x[0].shape[1:]))  # Print the total number of pixels in the first sample

    print(pix_acc(x, y, 3))  # Compute and print the pixel accuracy for the batch