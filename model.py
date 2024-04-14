import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class LeNet(nn.Module):
    def __init__(self, num_traits):
        super(LeNet, self).__init__()
        img_shape = 512
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_traits)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = torch.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_channel_in=3, class_num=None):
        super(ResNet, self).__init__()
    
        self.class_num = class_num
        self.c1 = nn.Conv2d(num_channel_in, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 6)


    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class AuxModel(nn.Module):
    def __init__(self, hidden=326, in_feat=163, class_num=None):
        super(AuxModel, self).__init__()
    
    
        self.class_num = class_num
        self.fc1 = torch.nn.Linear(in_feat, hidden)
        # print("fc1 dtype: " , self.fc1.weight.dtype)
        self.fc2 = torch.nn.Linear(hidden, 64)
        # print("fc2 dtype: " , self.fc2.weight.dtype)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x
    
    
        
    
class Ensemble(nn.Module):
    def __init__(self, image_model, aux_model):
        super(Ensemble, self).__init__()
        self.image_model = image_model

        self.aux_model = aux_model
        self.fc_mean = nn.LazyLinear(6)
        self.fc_sd = nn.LazyLinear(6)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1,x2):

        x1 = self.image_model(x1).float()
        x1 = self.dropout(x1)
        x2 = self.aux_model(x2).float()
        x = torch.cat((x1,x2), dim = 1)
        mean = self.fc_mean(x)
        # sd = torch.relu(self.fc_sd(x))
        # return (mean,sd)
        return mean
    
    

class EnsembleMulti(nn.Module):
    def __init__(self, image_model, model2, aux_model):
        super(EnsembleMulti, self).__init__()
        self.image_model = image_model
        self.model2 = model2
        self.aux_model = aux_model
        self.fc_mean = nn.LazyLinear(6)
        self.fc_sd = nn.LazyLinear(6)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1,x2):
        x3 = self.model2(x1).float()
        x1 = self.image_model(x1).float()
        # x1 = self.gap(x1)
        x1 = self.dropout(x1)
        x2 = self.aux_model(x2).float()
        x = torch.cat((x1,x2,x3), dim = 1)
        mean = self.fc_mean(x)
        sd = torch.relu(self.fc_sd(x))
        return (mean,sd)
    
    
