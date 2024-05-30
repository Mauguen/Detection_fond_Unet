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
        # Reshape logp for gather operation
        logp = logp.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # Gather log probabilities with respect to target
        logp = torch.gather(logp, 1, target.view(-1, 1)).view(n, H, W)

        # Define weights
        weight_class_0 = 1  # Poids pour la classe majoritaire
        weight_class_1 = H*W # Poids pour la classe minoritaire

        # Apply weights to the log probabilities
        weighted_logp = weight_class_0 * (target == 0).float() * logp + weight_class_1 * (target == 1).float() * logp

        # Compute loss
        loss = -weighted_logp.mean()
        return loss
