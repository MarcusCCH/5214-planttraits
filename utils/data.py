import pandas as pd
from tqdm import tqdm
from random import shuffle, seed
import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PlantDataset(Dataset):
    def __init__(self, train_csv, test_csv, image_dir, transform=None):
    
        self.train_df =pd.read_csv(train_csv)
        
        self.test_df = pd.read_csv(test_csv)
        classes = "X4_mean,X11_mean,X18_mean,X26_mean,X50_mean,X3112_mean".split(",")

        aux_classes = list(map(lambda x: x.replace("mean","sd"), classes))
        feature_cols = self.test_df.columns[1:-1].fillna(-1).tolist()

        self.train_features = self.train_df[feature_cols].values
        self.train_labels = self.train_df[classes].values
        self.train_aux_labels = self.train_df[aux_classes].values
        
        self.train_df['image_path'] = f'{image_dir}/train_images/'+self.train_df['id'].astype(str)+'.jpeg'
        self.images = (self.train_df[["image_path"]].values)
        self.id = self.train_df[["id"]].values
        # self.images = torch.tensor(list(map(lambda x: torchvision.io.read_image(x[0]), self.train_df[["image_path"]].values)))
        
        # print(self.images[0][0])
        # print(self.train_features[0])
        # print(self.train_labels[0])
        
    def __len__(self):
        return len(self.train_df) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torchvision.io.read_image(self.images[idx][0]).float()

        # if self.transform:
        #     sample = self.transform(sample)
        sample = {'id': self.id[idx][0],'images': image, 'features': torch.tensor(self.train_features[idx]).float(), 'labels': torch.tensor(self.train_labels[idx]).float(), 'aux_labels': torch.tensor(self.train_aux_labels[idx]).float()}
        # sample = {'images': torchvision.io.read_image(self.images[idx][0]), 'features': torch.tensor(self.train_features[idx]), 'labels': torch.tensor(self.train_labels[idx]), 'aux_labels': torch.tensor(self.train_aux_labels[idx])}

        return sample
    
