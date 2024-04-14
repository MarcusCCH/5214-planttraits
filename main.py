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

from utils.data import PlantDataset
from torcheval.metrics import R2Score

from model import LeNet, ResNet, AuxModel, Ensemble, EnsembleMulti
from torch.utils.data import random_split
from loss import  R2Loss, R2Metric
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_L_Weights, convnext_tiny, ConvNeXt_Small_Weights, vit_b_16, ViT_H_14_Weights, vit_b_32
import time
def add_parser_arguments(parser):
    parser.add_argument('--model', type=str, default="eff")
    parser.add_argument('--hidden', type=int, default=326)
    parser.add_argument('--num_traits', type=float, default=6)
    parser.add_argument('--loss_mean_weight', type=float, default=1.0)
    parser.add_argument('--loss_sd_weight', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)

    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
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
    with tqdm(train_loader, desc=f"train epoch {e}") as t:
            avg_loss = 0
            cnt = 0
            for i, (batch) in enumerate(t):
                # if i+1 == total_step: # no idea why loss explodes 
                #     break;
                images= batch["images"].to(device)
                features = batch["features"].to(device)

                labels = batch["labels"].to(device)
                # print(labels)
                aux_labels = batch["aux_labels"].to(device)

                # mean,sd = model(images, features)
                mean = model(images, features)

                # print(mean)
                # print(sd)
                
                
                optimizer.zero_grad()
                
                
                
                # loss = args.loss_mean_weight * R2Loss(labels,mean) + args.loss_sd_weight * R2Loss( aux_labels, sd)
                loss  = args.loss_mean_weight * R2Loss(labels,mean)
                # loss = args.loss_mean_weight * mean_loss + args.loss_sd_weight* sd_loss

                loss.backward()
                
                optimizer.step()
                
                time_elapsed = time.time() - start_time
                
                avg_loss += loss.item()
                cnt += 1
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Avg Loss: {:.4f}, Time used >>> {}:{}'.format(e+1, epochs, i+1, total_step, loss.item(), avg_loss/(i+1), int(time_elapsed / 60), int(time_elapsed % 60)))

            logging.info({"type": "train", "epoch": e, "loss" : avg_loss/cnt})

def eval_epoch(e, ds, model, device, name):
    model.eval()

    metric_data = defaultdict(list)

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers = 4)
    
    avg_loss = 0
    cnt = 0
    with tqdm(loader, desc=f"train e {e}", total=len(loader)) as t:
        for i, case in enumerate(t):

            images= case["images"].to(device)
            # images = images.unsqueeze(0).to(device)

            features = case["features"].to(device)
            # features = features.unsqueeze(0).to(device)
        

            labels = case["labels"].to(device)
            # aux_labels = case["aux_labels"].to(device)

            # mean,sd = model(images, features)
            mean = model(images, features)

            # r2  = R2Metric(labels,mean) 
            mean_metric = R2Score()
            mean_metric.update(mean, labels)
            r2 = mean_metric.compute()
            
            avg_loss += r2
            cnt += 1
            metric_data['epoch'].append(e)
            metric_data['id'].append(case["id"])
            metric_data['r2'].append(r2.detach().cpu().numpy())
        logging.info({"type": name, "epoch": e, "loss" : avg_loss/cnt})

    # for i, case in enumerate(ds):

    #     # images= case["images"].to(device)
    #     images = case["images"].unsqueeze(0).to(device)

    #     # features = case["features"].to(device)
    #     features = case["features"].unsqueeze(0).to(device)
    

    #     labels = case["labels"].unsqueeze(0).to(device)
    #     # aux_labels = case["aux_labels"].to(device)

    #     # mean,sd = model(images, features)
    #     mean = model(images, features)

    #     # r2  = R2Metric(labels,mean) 
    #     mean_metric = R2Score()
    #     mean_metric.update(mean, labels)
    #     r2 = mean_metric.compute()
            
    #     metric_data['epoch'].append(e)
    #     metric_data['id'].append(case["id"])
    #     metric_data['r2'].append(r2.detach().cpu().numpy())

    return metric_data    
    
if __name__ == "__main__":
    start_time = time.time()
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
    
    image_dir = "data/train_images"
    img_size = args.img_size
    ds = PlantDataset(train_csv, test_csv, image_dir, img_size)
    
    generator = torch.Generator().manual_seed(42)

    train_len = int(0.8 * len(ds)) + 2
    val_len = int(0.1 * len(ds))
    test_len = int(0.1 * len(ds))
    
    
    # train_ds, val_ds, test_ds = random_split(ds, [train_len,val_len,test_len ], generator = generator)
    train_ds, val_ds, test_ds = random_split(ds, [0.8,0.1,0.1 ], generator = generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, num_workers = 8)
    
    aux_model = AuxModel(hidden = args.hidden)
    if args.model.lower() == "multi":
        model = efficientnet_v2_s(weights=None)
        model1 = convnext_tiny(weights = None)
        model = EnsembleMulti(model, model1, aux_model)
    else:
        if args.model.lower() == "eff":
            model = efficientnet_v2_s(weights=None)
        elif args.model.lower() == "eff_l":
            model = efficientnet_v2_l(weights=None)
        elif args.model.lower() == "vit_b32":
            model = vit_b_32(weights=None) 
        elif args.model.lower() == "vit_b_16":
            model = vit_b_16(weights=None)    
        else:
            raise NotImplementedError()
        
        model = Ensemble(model,aux_model)

    
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
        

        # metric_data = eval_epoch(e,val_ds,  model, args.device)
        # pd.DataFrame(metric_data).to_csv(
        #     os.path.join(save_dir, f"valid_metric_{e}.csv"))
        
        # metric_data = eval_epoch(e,test_ds, model,args.device)
        # pd.DataFrame(metric_data).to_csv(
        #     os.path.join(save_dir, f"test_metric_{e}.csv"))
        
        train_epoch(e,train_loader, model, optimizer, args.device)
        if e % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{e}.pth"))
            
        # if e % args.eval_every == 0 or e == args.epochs:
        #     metric_data = eval_epoch(e,val_ds,  model, args.device, name="val")
        #     pd.DataFrame(metric_data).to_csv(
        #         os.path.join(save_dir, f"valid_metric_{e}.csv"))
            
        #     metric_data = eval_epoch(e,test_ds, model,args.device,name="test")
        #     pd.DataFrame(metric_data).to_csv(
        #         os.path.join(save_dir, f"test_metric_{e}.csv"))
            
        