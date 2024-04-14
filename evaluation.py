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

from model import LeNet, ResNet, AuxModel, Ensemble, EnsembleMulti
from torch.utils.data import random_split
from loss import  R2Loss, R2Metric, MultiTaskLossWrapper
from torchvision.models import resnet18, resnet50, efficientnet_v2_s, efficientnet_v2_l, EfficientNet_V2_L_Weights, convnext_tiny, ConvNeXt_Small_Weights, vit_b_32, vit_h_14, vit_b_16


def add_parser_arguments(parser):
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--hidden', type=str, default=326)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model', type=str, default = "eff")
    
def evaluate():
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)

    args = parser.parse_args()

    if not args.model_path and not args.model_dir:
        raise Exception("Need to supply model")
    
    model = args.model

    for root, dirnames, filenames in os.walk(args.model_dir):
        model_pths = [filename for filename in filenames if filename.endswith(".pth")]
            
    model_path = args.model_path or os.path.join(args.model_dir, model_pths[0]) # take custom path or the latest model
    print(f"Selected model: {model_path}")
                
    test_csv = "data/test.csv"
    image_dir = "data/test_images"
    
    ds = PlantDataset(test_csv, test_csv, image_dir)
 
    aux_model = AuxModel(hidden = args.hidden)
    if args.model.lower() == "multi":
        model = efficientnet_v2_s(weights=None)
        model1 = convnext_tiny(weights = None)
        model = EnsembleMulti(model, model1, aux_model)
    else:
        if args.model.lower() == "eff":
            model = efficientnet_v2_s(weights=None)
        elif args.model.lower() == "vit_b32":
            model = vit_b_32(weights=None) 
        elif args.model.lower() == "vit_b_16":
            model = vit_b_16(weights=None)    
        else:
            raise NotImplementedError()
        
        model = Ensemble(model,aux_model)

    device = args.device

    # model.load_state_dict(torch.load(args.model_path))
    model.load_state_dict(torch.load(model_path)) 
    model.to(device)    
    
    model.eval()

    preds = []
    test_loader= DataLoader(ds, batch_size=args.batch_size,
                         num_workers = 8)
    
 
    with tqdm(test_loader, desc="Evaluation: ") as t:
        for i, batch in enumerate(t):
            images= batch["images"].to(device)
            features = batch["features"].to(device)

            # mean,sd = model(images, features)        
            mean = model(images, features)        
            
            preds.extend(mean.detach().cpu().tolist())
            

            


    test_df = pd.read_csv("data/test.csv")
    pred_df = test_df[["id"]].copy()

    target_cols = "X4,X11,X18,X50,X26,X3112".split(",")
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