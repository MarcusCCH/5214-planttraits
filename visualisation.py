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

import jsons
from PIL import Image
import subprocess

def visualise():
    path = "output/train_full/test_metric_0.csv"
    image_path = "data/train_images/"
    vis_path = "visualisation/"
    df = pd.read_csv(path)
    filtered = df.query("r2<0")["id"].values
    r2 = df.query("r2<0")["r2"].values

    ids = []
    r2s = []
    
    for i,f in enumerate(filtered):
        f=f.replace("tensor([", "")
        f=f.replace("])", "")
        f=f.replace(" ", "")
        f=f.replace("\n", "")
        f = f.split(",")
        r2s.append(r2[i])
        ids.append(f)
        
    for idx, batch in enumerate(ids):
        folder = vis_path + str(r2s[idx])
        os.makedirs(folder, exist_ok = True)
        for b in batch:
            image = image_path + b + ".jpeg"
            subprocess.run(["cp", "-n", image, folder])


    

if __name__ == "__main__":
    visualise()
        
    