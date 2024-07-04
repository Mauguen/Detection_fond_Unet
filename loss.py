import torch
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np

class Dice_Loss(torch.nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def forward(self, pred, target):
        smooth = 0
        n, C, H, W = pred.shape

        # Apply sigmoid to predictions to get probabilities
        pred = torch.argmax(pred, dim=1).float()
        pred.requires_grad = True
        pred = torch.sigmoid(pred)
        weight_class_1 = H * W  # Poids pour la classe minoritaire
        weight_class_0 = 1  # Poids pour la classe majoritaire
        weights = weight_class_0 * (target == 0).float() + weight_class_1 * (target == 1).float()
        pred = pred * weights
        print(np.unique(pred.cpu().detach().numpy()), np.unique(target.cpu().detach().numpy()))

        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        # Compute Dice coefficient
        dice = (2. * intersection + smooth) / (union + smooth)

        # Dice Loss is 1 - Dice coefficient
        loss = 1 - dice

        return loss

class Binary_Cross_Entropy_Loss(torch.nn.Module):
    """Binary cross entropy loss."""

    def __init__(self):
        super(Binary_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target):
        n, C, H, W = pred.shape
        # Calculate log probabilities
        pred = torch.argmax(pred, dim=1).float()
        pred.requires_grad = True
        h_theta = torch.sigmoid(pred)
        # logp = F.logsigmoid(pred)

        # Define weights
        weight_class_0 = 1  # Poids pour la classe majoritaire
        weight_class_1 = H*W # Poids pour la classe minoritaire

        # Apply weights to the log probabilities
        # weighted_logp = weight_class_0 * (target == 0).float() * logp[:, 0] + weight_class_1 * (target == 1).float() * logp[:, 1]
        weighted_logp = weight_class_0 * (target == 0).float() * torch.log(1-h_theta) + weight_class_1 * (target == 1).float() * torch.log(h_theta)

        # Compute loss
        loss = -weighted_logp.mean()
        return loss
