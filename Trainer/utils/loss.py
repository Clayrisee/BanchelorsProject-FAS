from math import gamma
import torch
from torch import nn

class PixWiseBCELoss(nn.Module):
    def __init__(self,
    weight=None,
    beta=0.5):
        super().__init__()
        weight = torch.Tensor(1)
        self.criterion = nn.BCELoss(weight=weight) # for counter imbalanced dataset (idk it will be great or not)
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_loss = self.criterion(net_mask, target_mask)
        binary_loss = self.criterion(net_label, target_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss

class FocalBCELoss(nn.Module):
    """
    Implementation of FocalLoss using BCE.
    Reference:
    https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/18
    """
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, out, gt):
        bce_loss = self.criterion(out, gt)
        pt = torch.exp(-bce_loss)
        alpha_tensor = (1-self.alpha) + gt * (2 * self.alpha - 1) # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - pt) ** gamma * bce_loss
        return f_loss.mean()

class PixFocalLoss(nn.Module):
    def __init__(self, alpha=0.25,beta=0.5, gamma=2):
        super().__init__()
        self.criterion = FocalBCELoss(alpha, gamma)
        self.beta = beta
    
    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_focal_loss = self.criterion(net_mask, target_mask)
        binary_focal_loss = self.criterion(net_label, target_label)
        loss = pixel_focal_loss * self.beta + binary_focal_loss * (1 - self.beta)
        return loss