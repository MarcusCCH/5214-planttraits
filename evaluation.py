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

from model import LeNet, ResNet, AuxModel, Ensemble
from torch.utils.data import random_split
from loss import  R2Loss, R2Metric, MultiTaskLossWrapper
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, EfficientNet_V2_S_Weights

def add_parser_arguments(parser):
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=4)
    
def evaluate():
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    if not args.model_path:
        raise Exception("Need to supply model")
    
    test_csv = "data/test.csv"
    image_dir = "data/test_images"
    ds = PlantDataset(test_csv, test_csv, image_dir)
 
    aux_model = AuxModel()
    model = efficientnet_v2_s(weights=None)
    model = Ensemble(model, aux_model)
    
    model.load_state_dict(torch.load(args.model_path))
    device = args.device
    model.to(device)    
    
    model.eval()

    preds = []
    test_loader= DataLoader(ds, batch_size=args.batch_size,
                         num_workers = 4)
    
 
    with tqdm(test_loader, desc="Evaluation: ") as t:
        for i, batch in enumerate(t):
            images= batch["images"].to(device)
            features = batch["features"].to(device)

            mean,sd = model(images, features)        
            preds.extend(mean.detach().cpu().tolist())
            

            


    test_df = pd.read_csv("data/test.csv")
    # test_df = pd.read_csv("sample_test.csv")
    pred_df = test_df[["id"]].copy()

    target_cols = "X4,X11,X18,X26,X50,X3112".split(",")
    pred_df[target_cols] = preds

    print(pred_df)
    pred_df.to_csv("submission_1.csv", index=False)
    # sub_df = pd.read_csv(f'sample_submission.csv')
    # sub_df = sub_df[["id"]].copy()
    # sub_df = sub_df.merge(pred_df, on="id", how="left")

    # sub_df.to_csv("submission.csv", index=False)
    # sub_df.head()
    
    
    
if __name__ == "__main__":
    evaluate()