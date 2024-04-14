import torch
from torch import nn
from model import ResNet


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
        loss0 = self.L1(preds[0], targ[0])
        loss1 = self.L1(preds[1], targ[1])
        loss2 = self.L1(preds[2], targ[2])
        loss3 = self.L1(preds[3], targ[3])
        loss4 = self.L1(preds[4], targ[4])
        loss5 = self.L1(preds[5], targ[5])
        
        return loss0+loss1+loss2+loss3+loss4+loss5
