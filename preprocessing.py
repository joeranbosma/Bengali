#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:24:24 2020

@author: joeranbosma
"""
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import numpy as np
import os
import time, gc

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns
import cv2

from tensorflow.keras.metrics import categorical_accuracy, CategoricalAccuracy
import tensorflow.keras.backend as K

import wandb
from wandb.keras import WandbCallback
from starter_eda_model_funcs import get_model, resize, MultiOutputDataGenerator
from starter_eda_model_funcs import get_lr_reduction_calbacks
 
def perform_preprocessing(preprocess_args, data_path='Data/', prep_folder='prep/', out='parquet'):
    """Perform preprocessing and save results to folder.
    Parts: wether to save the result in parts (4), or a single parquet file.
    """
    prep_path = data_path + prep_folder
    train_df_ = pd.read_csv(f'{data_path}/train.csv')
    
    # Check whether preprocessing folder exists, else, make it.
    if not os.path.exists(prep_path):
        os.makedirs(prep_path)
    
    
    img_width, img_height = preprocess_args['image_width'], preprocess_args['image_height']
    # set warning if preprocessing is not finished, and the config and actual preprocessing
    # is thus not necessairily coupled correctly
    pd.Series({'WARING': "Preprocessing unfinished"}).to_csv(f"{prep_path}/config.csv", header=False)
    
    # perform resizing
    file_names = [f'{data_path}/train_image_data_{i}.parquet' for i in range(4)]
    perform_resize(file_names, prep_path=prep_path, target_height=img_height, target_width=img_width,
                   pad=preprocess_args['padding'], out=out)

    # save settings
    pd.Series(preprocess_args).to_csv(f"{prep_path}/config.csv", header=False)


# Adapted from source: https://www.kaggle.com/iafoss/image-preprocessing-128x128
def perform_resize(file_names, prep_path, orig_height=137, orig_width=236, target_height=64, target_width=64, pad=16, out='parquet'):
    x_tot,x2_tot = [],[]
    for i, fname in enumerate(file_names):
        df = pd.read_parquet(fname)
        #the input is inverted
        data = 255 - df.iloc[:, 1:].values.reshape(-1, orig_height, orig_width).astype(np.uint8)
        target_images = []
        for idx in tqdm(range(len(df)), desc=f'Part {i}'):
            name = df.iloc[idx,0]
            #normalize each image by its max val (colour intensity of signs wrt eachother)
            img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
            img = crop_resize(img, orig_height, orig_width, target_height, target_width, pad)
        
            x_tot.append((img/255.0).mean())
            x2_tot.append(((img/255.0)**2).mean())
            
            # check if writing to individual png files or parquet part files
            if out == 'png':
                 success = cv2.imwrite(f'{prep_path}/{name}.png', img)
#                 img_out.writestr(f'{prep_path}/{name}.png', img)
            else:
                 target_images.append(np.ravel(img)) # maybe need .copy() too
    
        # check if writing to parquet file
        if out == 'parquet':
            df_prep = pd.DataFrame(data=target_images, index=df.index)
            # parquet files require string columns
            df_prep.columns = np.array(list(df_prep.columns)).astype("str")
            df_prep.to_parquet(f'{prep_path}/part{i}.parquet')
    
    # calculate statistics of images
    img_mean = np.mean(x_tot)
    img_std = np.sqrt(np.mean(x2_tot) - img_mean**2)
    pd.Series(dict(
        img_mean = img_mean,
        img_std  = img_std,
    )).to_csv(f'{prep_path}/stats.csv', header=False)

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, orig_height, orig_width, target_height, target_width, pad):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides, they are removed
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80) #80 might be noise cut-off
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < orig_width - 13) else orig_width
    ymax = ymax + 10 if (ymax < orig_height - 10) else orig_height
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(target_width,target_height))
