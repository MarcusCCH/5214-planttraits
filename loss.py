import torch
from torch import nn
from model import ResNet

# class MultiTaskNet(nn.Module):
#     def __init__(self, input_dim, 
#                  output_dim_0 : int = 1, output_dim_1 : int = 3,
#                  hidden_dim : int = 200):
        
#         super(MultiTaskNet, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim_0 = output_dim_0
#         self.output_dim_1 = output_dim_1
#         self.hidden_dim = hidden_dim
        
#         self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
#         self.final_0 = nn.Linear(self.hidden_dim, self.output_dim_0)
#         self.final_1 = nn.Linear(self.hidden_dim, self.output_dim_1)     
        
#     def forward(self, x : torch.Tensor, task_id : int):
#         x = self.hidden(x)
#         x = torch.sigmoid(x)
#         if task_id == 0:
#             x = self.final_0(x)
#         elif task_id == 1:
#             x = self.final_1(x)
#         else:
#             assert False, 'Bad Task ID passed'
            
#         return x



def R2Loss(y_true, y_pred):
    mask = y_true == -1
    SS_res = torch.sum(torch.square((y_pred-y_true)), dim = 0)
    SS_total = torch.sum(torch.square((y_true- torch.mean(y_true, dim= 0))), dim = 0)
    # print("SSRES: ", SS_res)
    # print("SS_TOTAL: ", SS_total)
    eps = + 1e-6
    return torch.mean(SS_res / (SS_total + eps))

def R2Metric(y_true, y_pred):
    SS_res = torch.sum(torch.square((y_pred-y_true)), dim = 0)
    SS_total = torch.sum(torch.square((y_true- torch.mean(y_true, dim= 0))), dim = 0)

    eps = + 1e-6
    return torch.mean(1- SS_res / (SS_total + eps))
