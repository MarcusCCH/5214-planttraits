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

from utils.data import PlantDataset, CaseCollator

from model import LeNet, ResNet, AuxModel, Ensemble
from torch.utils.data import random_split
from loss import  R2Loss, R2Metric
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, EfficientNet_V2_S_Weights

def add_parser_arguments(parser):
    
    parser.add_argument('--num_traits', type=float, default=6)
    parser.add_argument('--loss_mean_weight', type=float, default=1.0)
    parser.add_argument('--loss_sd_weight', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)

    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--train_csv', type=str, default='sample.csv')
    parser.add_argument('--test_csv', type=str, default='data/test.csv')

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--case_name', type=str, default="test")
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')

def seed_everything(this_seed):
    seed(this_seed)
    np.random.seed(this_seed)
    torch.manual_seed(this_seed)
    torch.cuda.manual_seed(this_seed)
    torch.cuda.manual_seed_all(this_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(e, train_loader, model, optimizer, device):
    with tqdm(train_loader, desc=f"train e {e}") as t:
            for i, (batch) in enumerate(t):
                if i+1 == total_step:
                    break;
                images= batch["images"].to(device)
                features = batch["features"].to(device)

                labels = batch["labels"].to(device)
                aux_labels = batch["aux_labels"].to(device)

                mean,sd = model(images, features)

                # print(mean)
                # print(sd)
                
                
                optimizer.zero_grad()

                # loss = args.loss_mean_weight * R2Loss(labels,mean) + args.loss_sd_weight * R2Loss( aux_labels, sd)
                loss  = args.loss_mean_weight * R2Loss(labels,mean)


                loss.backward()
                
                optimizer.step()
                
                print('e [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(e+1, epochs, i+1, total_step, loss.item()))
                logging.info({"epoch": e, "loss" : loss.item()})

def eval_epoch(e, ds, model, device):
    model.eval()

    metric_data = defaultdict(list)

    with tqdm(ds, desc=f"train e {e}") as t:
        for i, case in enumerate(t):

            images= case["images"].to(device)
            images = images.unsqueeze(0)
            features = case["features"].to(device)
            features = features.unsqueeze(0)

            labels = case["labels"].to(device)
            aux_labels = case["aux_labels"].to(device)

            mean,sd = model(images, features)

            r2  = R2Metric(labels,mean)
            
            metric_data['epoch'].append(e)
            metric_data['id'].append(case["id"])
            metric_data['r2'].append(r2)

    return metric_data    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.case_name)

    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_dir, f"experiment.log"),
                        format='%(asctime)s %(message)s',
                        filemode='wt',
                        level=logging.INFO)

    # prerare dataset
    train_csv = args.train_csv
    test_csv = args.test_csv
    
    image_dir = "data"

    ds = PlantDataset(train_csv, test_csv, image_dir)
    
    generator = torch.Generator().manual_seed(42)

    train_len = int(0.8 * len(ds)) + 2
    val_len = int(0.1 * len(ds))
    test_len = int(0.1 * len(ds))
    
    
    train_ds, val_ds, test_ds = random_split(ds, [train_len,val_len,test_len ], generator = generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True)
    
    aux_model = AuxModel()
    
    # model = ResNet(3,6)
    # model = resnet50()
    # model = resnet50(weights=None, dropout = 0.5, num_classes=args.num_traits)
    # model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT, dropout = 0.5)
    model = efficientnet_v2_s(weights=None)
    # model = efficientnet_v2_s(weights=None, dropout = 0.5, num_classes=args.num_traits)

    model = Ensemble(model, aux_model)
    
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
    
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    epochs = args.epochs


    model.train()
    total_step = len(train_loader)
    
    print("start training >>>")
    for e in range(epochs):  
        train_epoch(e,train_loader, model, optimizer, args.device)
        
        # if e % args.eval_every == 0 or e == args.epochs:
        # metric_data = eval_epoch(e,val_ds,  model, args.device)
        # print(metric_data)
        # pd.DataFrame(metric_data).to_csv(
        #     os.path.join(save_dir, f"valid_metric_{e}.csv"))
        
        # metric_data = eval_epoch(e,test_ds, model,args.device)
        # pd.DataFrame(metric_data).to_csv(
        #     os.path.join(save_dir, f"test_metric_{e}.csv"))
            
        if e % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{e}.pth"))