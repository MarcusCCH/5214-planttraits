import pandas as pd
from tqdm import tqdm
from random import shuffle, seed
import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class PlantDataset(Dataset):
    def __init__(self, train_csv, test_csv, image_dir, transform=None):
    
        self.df =pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        
        feature_cols = self.test_df.columns[1:-1].tolist()
        self.train_features = self.df[feature_cols].values
        
        self.df['image_path'] = f'{image_dir}/'+self.df['id'].astype(str)+'.jpeg'
        self.images = (self.df[["image_path"]].values)
        self.id = self.df[["id"]].values


        self.is_train = False
        
        if "train" in train_csv:
            self.is_train = True
            classes = "X4_mean,X11_mean,X18_mean,X26_mean,X50_mean,X3112_mean".split(",")
            aux_classes = list(map(lambda x: x.replace("mean","sd"), classes))  
            self.train_labels = self.df[classes].values
            self.train_aux_labels = self.df[aux_classes].values
            self.transform = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomAdjustSharpness(p=0.5, sharpness_factor=0.8),
            v2.RandomSolarize(p=0.5, threshold=0.5),
            v2.RandomGrayscale(p=0.1)
            ])
    
        # print(self.images[0][0])
        # print(self.train_features[0])
        # print(self.train_labels[0])
        
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torchvision.io.read_image(self.images[idx][0])
        
        if self.is_train:
            if self.transform: 
                image = self.transform(image)
                sample = {'id': self.id[idx][0],'images': image.float(), 'features': torch.tensor(self.train_features[idx]).float(), 'labels': torch.tensor(self.train_labels[idx]).float(), 'aux_labels': torch.tensor(self.train_aux_labels[idx]).float()}
        else:
            sample = {'id': self.id[idx][0],'images': image.float(), 'features': torch.tensor(self.train_features[idx]).float()}


        return sample
    
