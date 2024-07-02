import torch
from torch.nn import functional as F
from torch.autograd import Function
import numpy as np
    
class Binary_Cross_Entropy_Loss(torch.nn.Module):
    """Binary cross entropy loss."""

    def __init__(self):
        super(Binary_Cross_Entropy_Loss, self).__init__()

    def forward(self, pred, target):
        n, C, H, W = pred.shape
        # Calculate log probabilities
        logp = F.logsigmoid(pred)

        # Define weights
        weight_class_0 = 1  # Poids pour la classe majoritaire
        weight_class_1 = H*W # Poids pour la classe minoritaire

        # Apply weights to the log probabilities
        weighted_logp = weight_class_0 * (target == 0).float() * logp[:, 0] + weight_class_1 * (target == 1).float() * logp[:, 1]

        # Compute loss
        loss = -weighted_logp.mean()
        return loss
