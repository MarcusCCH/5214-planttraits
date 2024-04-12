import pandas as pd
from tqdm import tqdm
from random import shuffle, seed
import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_data(train_path, test_path, image_path):
    assert "csv" in train_path
    try:
        train_df =pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except:
        raise Exception("Error loading training data at path: {train_path}")

    cases = []

    classes = "X4_mean,X11_mean,X18_mean,X26_mean,X50_mean,X3112_mean".split(",")

    aux_classes = list(map(lambda x: x.replace("mean","sd"), classes))
    feature_cols = test_df.columns[1:-1].tolist()

    train_features = train_df[feature_cols].values
    train_labels = train_df[classes].values
    train_aux_labels = train_df[aux_classes].values
    
    train_df['image_path'] = f'{image_path}/train_images/'+train_df['id'].astype(str)+'.jpeg'
    images = train_df[["image_path"]].values
    
    # print(images[:2])
    
    # print(train_features[:2])
    # print(train_labels[:2])
    # print(train_aux_labels[:2])

    # for case in tqdm(df.iterrows()):
    #     data = {}
    #     for attr in attr_list: 
    #         data[attr] = case[1][attr]
    #     case = PlantCase(data, image_path)
    #     cases.append(case)
    

    shuffle(cases)

    return cases

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
    
# class PlantCase:
#     def __init__(self, data, image_path):

#         self.id =int(data["id"])
#         self.image_url = os.path.join(image_path, f"{self.id}.jpeg")
#         self.image =  torchvision.io.read_image(self.image_url)

#         self.trait_mean = [v for index,(k,v) in enumerate(data.items()) if "mean" in k]
#         self.trait_sd = [v for index,(k,v) in enumerate(data.items()) if "sd" in k]   

        
#     def pprint(self):

#         # print(self.image)
#         print(self.trait_mean)
#         print(self.trait_sd)
        
class CaseCollator:
    def __call__(self, cases):
        # print(len(cases))
        # print(type(cases))
        imgs = []
        mean = []
        sd = []
        for case in cases:
            img = case.image.cpu().detach().numpy()
            imgs.append(img)
            mean.append(case.trait_mean)
            sd.append(case.trait_sd)
  
        return (torch.Tensor(np.array(imgs)), torch.Tensor(mean),torch.Tensor(sd))
        # return (np.array(imgs), torch.Tensor(mean),torch.Tensor(sd))

        
           

if __name__ == "__main__":
    # list = "WORLDCLIM_BIO1_annual_mean_temperature,WORLDCLIM_BIO12_annual_precipitation,WORLDCLIM_BIO13.BIO14_delta_precipitation_of_wettest_and_dryest_month,WORLDCLIM_BIO15_precipitation_seasonality,WORLDCLIM_BIO4_temperature_seasonality,WORLDCLIM_BIO7_temperature_annual_range,SOIL_bdod_0.5cm_mean_0.01_deg,SOIL_bdod_100.200cm_mean_0.01_deg,SOIL_bdod_15.30cm_mean_0.01_deg,SOIL_bdod_30.60cm_mean_0.01_deg,SOIL_bdod_5.15cm_mean_0.01_deg,SOIL_bdod_60.100cm_mean_0.01_deg,SOIL_cec_0.5cm_mean_0.01_deg,SOIL_cec_100.200cm_mean_0.01_deg,SOIL_cec_15.30cm_mean_0.01_deg,SOIL_cec_30.60cm_mean_0.01_deg,SOIL_cec_5.15cm_mean_0.01_deg,SOIL_cec_60.100cm_mean_0.01_deg,SOIL_cfvo_0.5cm_mean_0.01_deg,SOIL_cfvo_100.200cm_mean_0.01_deg,SOIL_cfvo_15.30cm_mean_0.01_deg,SOIL_cfvo_30.60cm_mean_0.01_deg,SOIL_cfvo_5.15cm_mean_0.01_deg,SOIL_cfvo_60.100cm_mean_0.01_deg,SOIL_clay_0.5cm_mean_0.01_deg,SOIL_clay_100.200cm_mean_0.01_deg,SOIL_clay_15.30cm_mean_0.01_deg,SOIL_clay_30.60cm_mean_0.01_deg,SOIL_clay_5.15cm_mean_0.01_deg,SOIL_clay_60.100cm_mean_0.01_deg,SOIL_nitrogen_0.5cm_mean_0.01_deg,SOIL_nitrogen_100.200cm_mean_0.01_deg,SOIL_nitrogen_15.30cm_mean_0.01_deg,SOIL_nitrogen_30.60cm_mean_0.01_deg,SOIL_nitrogen_5.15cm_mean_0.01_deg,SOIL_nitrogen_60.100cm_mean_0.01_deg,SOIL_ocd_0.5cm_mean_0.01_deg,SOIL_ocd_100.200cm_mean_0.01_deg,SOIL_ocd_15.30cm_mean_0.01_deg,SOIL_ocd_30.60cm_mean_0.01_deg,SOIL_ocd_5.15cm_mean_0.01_deg,SOIL_ocd_60.100cm_mean_0.01_deg,SOIL_ocs_0.30cm_mean_0.01_deg,SOIL_phh2o_0.5cm_mean_0.01_deg,SOIL_phh2o_100.200cm_mean_0.01_deg,SOIL_phh2o_15.30cm_mean_0.01_deg,SOIL_phh2o_30.60cm_mean_0.01_deg,SOIL_phh2o_5.15cm_mean_0.01_deg,SOIL_phh2o_60.100cm_mean_0.01_deg,SOIL_sand_0.5cm_mean_0.01_deg,SOIL_sand_100.200cm_mean_0.01_deg,SOIL_sand_15.30cm_mean_0.01_deg,SOIL_sand_30.60cm_mean_0.01_deg,SOIL_sand_5.15cm_mean_0.01_deg,SOIL_sand_60.100cm_mean_0.01_deg,SOIL_silt_0.5cm_mean_0.01_deg,SOIL_silt_100.200cm_mean_0.01_deg,SOIL_silt_15.30cm_mean_0.01_deg,SOIL_silt_30.60cm_mean_0.01_deg,SOIL_silt_5.15cm_mean_0.01_deg,SOIL_silt_60.100cm_mean_0.01_deg,SOIL_soc_0.5cm_mean_0.01_deg,SOIL_soc_100.200cm_mean_0.01_deg,SOIL_soc_15.30cm_mean_0.01_deg,SOIL_soc_30.60cm_mean_0.01_deg,SOIL_soc_5.15cm_mean_0.01_deg,SOIL_soc_60.100cm_mean_0.01_deg,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m1,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m1,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m1,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m1,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m1,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m10,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m10,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m10,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m10,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m10,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m11,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m11,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m11,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m11,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m11,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m12,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m12,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m12,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m12,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m12,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m2,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m2,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m2,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m2,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m2,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m3,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m3,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m3,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m3,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m3,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m4,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m4,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m4,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m4,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m4,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m5,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m5,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m5,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m5,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m5,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m6,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m6,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m6,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m6,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m6,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m7,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m7,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m7,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m7,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m7,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m8,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m8,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m8,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m8,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m8,MODIS_2000.2020_monthly_mean_surface_reflectance_band_01_._month_m9,MODIS_2000.2020_monthly_mean_surface_reflectance_band_02_._month_m9,MODIS_2000.2020_monthly_mean_surface_reflectance_band_03_._month_m9,MODIS_2000.2020_monthly_mean_surface_reflectance_band_04_._month_m9,MODIS_2000.2020_monthly_mean_surface_reflectance_band_05_._month_m9,VOD_C_2002_2018_multiyear_mean_m01,VOD_C_2002_2018_multiyear_mean_m02,VOD_C_2002_2018_multiyear_mean_m03,VOD_C_2002_2018_multiyear_mean_m04,VOD_C_2002_2018_multiyear_mean_m05,VOD_C_2002_2018_multiyear_mean_m06,VOD_C_2002_2018_multiyear_mean_m07,VOD_C_2002_2018_multiyear_mean_m08,VOD_C_2002_2018_multiyear_mean_m09,VOD_C_2002_2018_multiyear_mean_m10,VOD_C_2002_2018_multiyear_mean_m11,VOD_C_2002_2018_multiyear_mean_m12,VOD_Ku_1987_2017_multiyear_mean_m01,VOD_Ku_1987_2017_multiyear_mean_m02,VOD_Ku_1987_2017_multiyear_mean_m03,VOD_Ku_1987_2017_multiyear_mean_m04,VOD_Ku_1987_2017_multiyear_mean_m05,VOD_Ku_1987_2017_multiyear_mean_m06,VOD_Ku_1987_2017_multiyear_mean_m07,VOD_Ku_1987_2017_multiyear_mean_m08,VOD_Ku_1987_2017_multiyear_mean_m09,VOD_Ku_1987_2017_multiyear_mean_m10,VOD_Ku_1987_2017_multiyear_mean_m11,VOD_Ku_1987_2017_multiyear_mean_m12,VOD_X_1997_2018_multiyear_mean_m01,VOD_X_1997_2018_multiyear_mean_m02,VOD_X_1997_2018_multiyear_mean_m03,VOD_X_1997_2018_multiyear_mean_m04,VOD_X_1997_2018_multiyear_mean_m05,VOD_X_1997_2018_multiyear_mean_m06,VOD_X_1997_2018_multiyear_mean_m07,VOD_X_1997_2018_multiyear_mean_m08,VOD_X_1997_2018_multiyear_mean_m09,VOD_X_1997_2018_multiyear_mean_m10,VOD_X_1997_2018_multiyear_mean_m11,VOD_X_1997_2018_multiyear_mean_m12"
    # print(list.split(","))
    
    get_training_data("sample.csv", "data/test_images")