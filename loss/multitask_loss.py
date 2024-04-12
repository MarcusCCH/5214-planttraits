import torch
from torch import nn

class MultiTask_Network(nn.Module):
    def __init__(self, input_dim, 
                 output_dim_0 : int = 1, output_dim_1 : int = 3,
                 hidden_dim : int = 200):
        
        super(MultiTask_Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim_0 = output_dim_0
        self.output_dim_1 = output_dim_1
        self.hidden_dim = hidden_dim
        
        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.final_0 = nn.Linear(self.hidden_dim, self.output_dim_0)
        self.final_1 = nn.Linear(self.hidden_dim, self.output_dim_1)     
        
    def forward(self, x : torch.Tensor, task_id : int):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        if task_id == 0:
            x = self.final_0(x)
        elif task_id == 1:
            x = self.final_1(x)
        else:
            assert False, 'Bad Task ID passed'
            
        return x