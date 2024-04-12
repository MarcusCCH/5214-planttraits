import torch
from torch import nn
from model import ResNet
import numpy as np

def R2Loss(y_true, y_pred):
    SS_res = torch.sum(torch.square((y_pred-y_true)), dim = 0)
    SS_total = torch.sum(torch.square((y_true- torch.mean(y_true, dim= 0))), dim = 0)
    eps = + 1e-6
    return torch.mean(SS_res / (SS_total + eps))

def R2Metric(y_true, y_pred):
    SS_res = torch.sum(torch.square((y_pred-y_true)), dim = 0)
    SS_total = torch.sum(torch.square((y_true- torch.mean(y_true, dim= 0))), dim = 0)

    eps = + 1e-6
    return torch.mean(1- SS_res / (SS_total + eps))


class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.L1Loss()

    def forward(self, preds, targ):
        total_loss = 0
        for b in range(len(preds)):
            for i,(p,t) in enumerate(zip(preds[b],targ[b])):
                total_loss += self.L1(p,t)
     
        return total_loss
