import argparse
from random import shuffle, seed
import os
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.data import get_training_data, CaseCollator

from model import LeNet, ResNet

from loss.multitask_loss import MultiTaskLoss
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, EfficientNet_V2_S_Weights

def add_parser_arguments(parser):
    
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--csv_path', type=str, default='sample.csv')

    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--case_name', type=str, default="test")
    parser.add_argument('--num_traits', type=float, default=6)
    parser.add_argument('--seed', type=int, default=42)

def seed_everything(this_seed):
    seed(this_seed)
    np.random.seed(this_seed)
    torch.manual_seed(this_seed)
    torch.cuda.manual_seed(this_seed)
    torch.cuda.manual_seed_all(this_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    seed_everything(args.seed)

    save_dir = os.path.join(args.save_dir, args.case_name)

    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_dir, f"experiment.log"),
                        format='%(asctime)s %(message)s',
                        filemode='wt',
                        level=logging.INFO)

    # prerare dataset
    # train_csv = os.path.join(args.data_dir, "train.csv")
    train_csv = args.csv_path
    train_images = "data/train_images"
    training_data = get_training_data(train_csv, train_images)

    total_num = len(training_data)
    train_num = int(total_num * args.train_ratio) + 1
    valid_num = int(total_num * args.valid_ratio)

    train_cases = training_data[:train_num]
    valid_cases = training_data[train_num: train_num+valid_num]
    test_cases  = training_data[train_num+valid_num:]

    print("number of training case", len(train_cases))
    print("number of validation case", len(valid_cases))
    print("number of testing case", len(test_cases))

    train_loader = DataLoader(train_cases,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CaseCollator())

    print("prepare train loader")

    # model = ResNet(3,6)
    # model = resnet50()
    # model = resnet50(weights=None, dropout = 0.5, num_classes=args.num_traits)
    # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT, dropout = 0.5, num_classes=args.num_traits)
    model = efficientnet_v2_s(weights=None, dropout = 0.5, num_classes=args.num_traits)
    model.to(args.device)

    is_regression = torch.Tensor([1,1,1,1,1,1])
    multitaskloss_instance = MultiTaskLoss(is_regression, reduction = "sum")
    
    params = list(model.parameters()) + list(multitaskloss_instance.parameters())


    optimizer = torch.optim.Adam(params,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    epochs = args.epochs
    total_step = len(train_loader)

    model.train()
    multitaskloss_instance.train()

    for epoch in range(epochs):  
        with tqdm(train_loader, desc=f"train epoch {epoch}") as t:
            for i, (images, mean,sd) in enumerate(t):
                model.train()

                images= images.to(args.device)
                mean = mean.to(args.device)
                sd = sd.to(args.device)

                outputs = model(images)
                
                optimizer.zero_grad()


                # loss = criterion(outputs, mean)
                loss = (outputs - mean)
                loss = loss.mean(dim=0)
                loss.retain_grad()
                # loss.backward()
                # multitaskloss = multitaskloss_instance(loss)
                
                # loss.backward(gradient = torch.ones(args.num_traits).to(args.device))
                # multitaskloss.backward()

                for idx, loss_i in enumerate(loss):
                    # print(loss_i)
                    loss_i.backward(retain_graph=True)
                optimizer.step()
                
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.sum().item()))

  