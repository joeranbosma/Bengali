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
 
def perform_preprocessing(preprocess_args, data_path='Data/', prep_folder='prep/'):
    """Perform preprocessing and save results to folder.
    Parts: wether to save the result in parts (4), or a single parquet file.
    """
	prep_path = data_path + prep_folder
	train_df_ = pd.read_csv(f'{data_path}/train.csv')
	
	# Check whether preprocessing folder exists, else, make it.
	if not os.path.exists(prep_path):
		os.path.mkdirs(prep_path)
			
	
    IMG_SIZE = preprocess_args['image_size']
    # set warning if preprocessing is not finished, and the config and actual preprocessing
    # is thus not necessairily coupled correctly
    pd.Series({'WARING': "Preprocessing unfinished"}).to_csv(f"{prep_path}/config.csv", header=False)
    
    for i in range(4):
        print(f"STEP {i+1}/4")
        train_df = pd.merge(pd.read_parquet(f'{data_path}/train_image_data_{i}.parquet'), 
                            train_df_, on='image_id')
        
        # drop target labels and 'move' image_id information to the index (so that you can do graphic modifications without labeldata)
        train_df.drop(columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], inplace=True)
        train_df.set_index('image_id', inplace=True)
        # resize images
        X_train = resize_padding(train_df, resize_size=IMG_SIZE, padding=preprocess_args['padding'])/255
        del train_df
        
		# save X_train
		X_train.columns = np.array(list(X_train.columns)).astype("str")
		X_train.to_parquet(f"{prep_path}/part{i}.parquet", index=True)

        del X_train			
			
    # save settings
    pd.Series(preprocess_args).to_csv(f"{prep_path}/config.csv")
	
	
# Adapted from source: https://www.kaggle.com/iafoss/image-preprocessing-128x128
def perform_resize(file_names, prep_path, orig_height=137,orig_width=236,target_height=64,target_width=64, pad=16):
	x_tot,x2_tot = [],[]
	for i, fname in enumerate(file_names):
		df = pd.read_parquet(fname)
		#the input is inverted
		data = 255 - df.iloc[:, 1:].values.reshape(-1, orig_height, orig_width).astype(np.uint8)
		target_images = []
		for idx in tqdm(range(len(df))):
			name = df.iloc[idx,0]
			#normalize each image by its max val (colour intensity of signs wrt eachother)
			img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
			img = crop_resize(img, orig_height, orig_width, target_height, target_width, pad)
		
			x_tot.append((img/255.0).mean())
			x2_tot.append(((img/255.0)**2).mean())
			target_images.append(img.copy())
			# img = cv2.imencode('.png',img)[1]
			# img_out.writestr(name + '.png', img)
		
		df = pd.DataFrame(data=target_images, index=data.index)
		df.to_parquet(f'{prep_path}/part{i}.parquet')
	
	# calculate statistics of images
	img_mean = np.mean(x_tot)
	img_std = np.sqrt(np.mean(x2_tot) - img_mean**2)
	pd.Series(dict(
		img_mean = img_mean,
		img_std  = img_std,
	)).to_csv(f'{prep_path}/stats.csv')

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