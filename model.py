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
    def __init__(self, in_feat=162, class_num=None):
        super(AuxModel, self).__init__()
    
    
        self.class_num = class_num
        self.fc1 = torch.nn.Linear(in_feat, 326)
        # print("fc1 dtype: " , self.fc1.weight.dtype)
        self.fc2 = torch.nn.Linear(326, 64)
        # print("fc2 dtype: " , self.fc2.weight.dtype)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class MultiTaskLayer(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.linear = nn.LazyLinear(out_features)
        self.final = nn.Linear(out_features, 1)
    def forward(self, x):
        # h0 = self.linear(x)
        # h1 = self.linear(x)
        # h2 = self.linear(x)
        # h3 = self.linear(x)
        # h4 = self.linear(x)
        # h5 = self.linear(x)
        # SSD = self.final(h0)
        # SLA = self.final(h1)
        # GH = self.final(h2)
        # SM = self.final(h3)
        # LN = self.final(h4)
        # LA = self.final(h5)
        out = None
        for _ in range(6):
            t = self.final(self.linear(x))
            if out is None:
                out = t
            else:
                out = torch.hstack((out,t))
        assert out.size(dim=1) == 6 # check if number of traits == 6
        return out
    
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
        # x1 = self.gap(x1)
        x1 = self.dropout(x1)
        x2 = self.aux_model(x2).float()
        x = torch.cat((x1,x2), dim = 1)
        mean = self.fc_mean(x)
        sd = self.fc_sd(x)
        return (mean,sd)
    
    
class EnsembleMultiTask(nn.Module):
    def __init__(self, image_model, aux_model):
        super(EnsembleMultiTask, self).__init__()
        self.image_model = image_model
        self.aux_model = aux_model
        self.fc_mean = MultiTaskLayer(512)
        self.fc_sd = MultiTaskLayer(512)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1,x2):
        x1 = self.image_model(x1).float()
        # x1 = self.gap(x1)
        x1 = self.dropout(x1)
        x2 = self.aux_model(x2).float()
        x = torch.cat((x1,x2), dim = 1)
        pred_mean = self.fc_mean(x)
        # sd = self.fc_sd(x)
        return pred_mean
    
    
