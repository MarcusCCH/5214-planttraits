import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, num_channel_in=16*3, class_num=None):
        super(ResNet, self).__init__()
        self.c1 = nn.Conv2d(num_channel_in, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.c3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(256, class_num)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        # x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
