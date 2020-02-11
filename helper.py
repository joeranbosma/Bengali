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


# adapted from https://github.com/keras-team/keras/issues/4506
class GlobalAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.accs = []

    def eval_acc(self):
        x_val, y_true = self.validation_data
        y_pred = self.model.predict(x_val)
        global_acc = 0
        global_weight = [0.5, 0.25, 0.25]
        assert np.sum(global_weight) == 1., "Set weights to sum to one, for normalization"
        for i in range(3):
            acc = K.mean(categorical_accuracy(y_true[i], y_pred[i]))
            global_acc += acc * global_weight[i]
        
        return global_acc

    def on_epoch_end(self, epoch, logs={}):
        acc = self.eval_acc()
        # print("Global accuracy for epoch %d is %f"%(epoch, acc))
        self.accs.append(acc)
        wandb.log({'val_global_accuracy': acc}, step=wandb.run.step)

# adapted from https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
# added WandB integration and custom datagen arguments
def train(train_df_, datagen_args, name=None, epochs=30, model=None,
          N_CHANNELS=1, batch_size=256, preprocess_args={'image_size': 64, 'padding': 0}):
    IMG_SIZE = preprocess_args['image_size']
    
    # set up config
    config = datagen_args.copy()
    config['epochs'] = epochs
    config['name'] = name
    config.update(preprocess_args)
    wandb.init(project='mlip', name=name, config=config)

    if model == None:
        model = get_model(img_size=IMG_SIZE)

    histories = []
    for i in range(4):
        train_df = pd.merge(pd.read_parquet(f'Data/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)
        
        X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
		# Stores it as float32 (but starts as int8, which is more memory efficient)
		# This is a normalization, which you can also do at the end, so that the 
		# processing done after this line does not work with float32 but with int8
		# 255 = scale colour range to 0 to 1.
		# Preprocessing 4 blocks takes approx 4x1 minute
        X_train = resize_padding(X_train, resize_size=IMG_SIZE, padding=preprocess_args['padding'])/255
        
        # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
        X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
        
        Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
        Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
        Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values

        print(f'Training images: {X_train.shape}')
        print(f'Training labels root: {Y_train_root.shape}')
        print(f'Training labels vowel: {Y_train_vowel.shape}')
        print(f'Training labels consonants: {Y_train_consonant.shape}')

        # Divide the data into training and validation set
        x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
        del train_df
        del X_train
        del Y_train_root, Y_train_vowel, Y_train_consonant

        datagen = MultiOutputDataGenerator(**datagen_args)
        # This will just calculate parameters required to augment the given data. This won't perform any augmentations
        datagen.fit(x_train)
        
        # Visualize few samples of current training dataset, including data augmentation
        preview_data_aug(datagen, x_train, {'out_root': y_train_root, 
                                           'out_vowel': y_train_vowel, 
                                           'out_consonant': y_train_consonant}, IMG_SIZE=IMG_SIZE)

        # create custom global accuracy with weights 50%, 25%, 25%
        global_accuracy_callback = GlobalAccuracyCallback(
            validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]))
        
        # get lr reduction on plateau callbacks
        lr_reduction_root, lr_reduction_vowel, lr_reduction_consonant = get_lr_reduction_calbacks()
        
        # Fit the model
        history = model.fit(datagen.flow(x_train, {'out_root': y_train_root, 
                                                   'out_vowel': y_train_vowel, 
                                                   'out_consonant': y_train_consonant}, 
                                         batch_size=batch_size),
                                epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                                steps_per_epoch=x_train.shape[0] // batch_size, 
                                callbacks=[lr_reduction_root, lr_reduction_vowel, 
                                            lr_reduction_consonant, WandbCallback(),
                                           global_accuracy_callback])

        histories.append(history.history)
        
        # save model online
        model.save(os.path.join(wandb.run.dir, "model_{}.h5".format(i)))
        np.save(os.path.join(wandb.run.dir, "histories.npy"), histories)
        
        # Delete to reduce memory usage
        del x_train
        del x_test
        del y_train_root
        del y_test_root
        del y_train_vowel
        del y_test_vowel
        del y_train_consonant
        del y_test_consonant
        gc.collect()

def train_from_prep(train_df_, datagen_args, name=None, epochs=30, model=None,
          N_CHANNELS=1, batch_size=256, preprocess_args={'image_size': 64, 'padding': 0},
          data_path='Data/', prep_folder='prep/', parts=True):
    IMG_SIZE = preprocess_args['image_size']
    
    # set up config
    config = datagen_args.copy()
    config['epochs'] = epochs
    config['name'] = name
    config.update(preprocess_args)
    wandb.init(project='mlip', name=name, config=config)

    if model == None:
        model = get_model(img_size=IMG_SIZE)
    
    histories = []
    for i in range(4):
        train_df = pd.merge(pd.read_parquet(f'{data_path}/{prep_folder}/part{i}.parquet'), train_df_, left_index=True, right_on='image_id').drop(['image_id'], axis=1)
        
        # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
        X_train = train_df.drop(columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
        
        Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
        Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
        Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values

        print(f'Training images: {X_train.shape}')
        print(f'Training labels root: {Y_train_root.shape}')
        print(f'Training labels vowel: {Y_train_vowel.shape}')
        print(f'Training labels consonants: {Y_train_consonant.shape}')

        # Divide the data into training and validation set
        x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
        del train_df
        del X_train
        del Y_train_root, Y_train_vowel, Y_train_consonant

        datagen = MultiOutputDataGenerator(**datagen_args)
        # This will just calculate parameters required to augment the given data. This won't perform any augmentations
        datagen.fit(x_train)
        
        # Visualize few samples of current training dataset, including data augmentation
        preview_data_aug(datagen, x_train, {'out_root': y_train_root, 
                                           'out_vowel': y_train_vowel, 
                                           'out_consonant': y_train_consonant}, IMG_SIZE=IMG_SIZE)

        # create custom global accuracy with weights 50%, 25%, 25%
        global_accuracy_callback = GlobalAccuracyCallback(
            validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]))
        
        # get lr reduction on plateau callbacks
        lr_reduction_root, lr_reduction_vowel, lr_reduction_consonant = get_lr_reduction_calbacks()
        
        # Fit the model
        history = model.fit(datagen.flow(x_train, {'out_root': y_train_root, 
                                                   'out_vowel': y_train_vowel, 
                                                   'out_consonant': y_train_consonant}, 
                                         batch_size=batch_size),
                                epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                                steps_per_epoch=x_train.shape[0] // batch_size, 
                                callbacks=[lr_reduction_root, lr_reduction_vowel, 
                                            lr_reduction_consonant, WandbCallback(),
                                           global_accuracy_callback])

        histories.append(history.history)
        
        # save model online
        model.save(os.path.join(wandb.run.dir, "model_{}.h5".format(i)))
        np.save(os.path.join(wandb.run.dir, "histories.npy"), histories)
        
        # Delete to reduce memory usage
        del x_train
        del x_test
        del y_train_root
        del y_test_root
        del y_train_vowel
        del y_test_vowel
        del y_train_consonant
        del y_test_consonant
        gc.collect()

def resize_padding(df, resize_size=64, padding=0, need_progress_bar=True):
    """
    Graphemes are resized to the roi, plus `padding` number of pixels in all directions.
    The resize_size is the size of the final image, so to obtain an 'unpadded' image 
    of 64x64 with a wiggle room of 4 pixels, set resize_size to 72 and padding to 4. 
    """
    resized = {}
    
    iterator = range(df.shape[0])
    if need_progress_bar: iterator = tqdm(iterator)

    for i in iterator:
        image=df.loc[df.index[i]].values.reshape(137,236)
        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        idx = 0 
        ls_xmin = []
        ls_ymin = []
        ls_xmax = []
        ls_ymax = []
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)
        
        ymin = max(ymin-padding, 0)
        ymax = min(ymax+padding, 137)
        xmin = max(xmin-padding, 0)
        xmax = min(xmax+padding, 236)
        roi = image[ymin:ymax,xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        resized[df.index[i]] = resized_roi.reshape(-1)
    
    resized = pd.DataFrame(resized).T
    return resized

def perform_preprocessing(train_df_, preprocess_args, data_path='Data/', prep_folder='prep/', parts=False):
    """Perform preprocessing and save results to folder.
    Parts: wether to save the result in parts (4), or a single parquet file.
    """
    IMG_SIZE = preprocess_args['image_size']
    # set warning if preprocessing is not finished, and the config and actual preprocessing
    # is thus not necessairily coupled correctly
    pd.Series({'WARING': "Preprocessing unfinished"}).to_csv(f"{data_path}/{prep_folder}/config.csv", header=False)
    
    for i in range(4):
        print(f"STEP {i+1}/4")
        train_df = pd.merge(pd.read_parquet(f'{data_path}/train_image_data_{i}.parquet'), 
                            train_df_, on='image_id')
        
        # drop target labels and 'move' image_id information to the index
        train_df.drop(columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], inplace=True)
        train_df.set_index('image_id', inplace=True)
        # resize images
        X_train = resize_padding(train_df, resize_size=IMG_SIZE, padding=preprocess_args['padding'])/255
        del train_df
        
        if parts:
            # save X_train
            X_train.columns = np.array(list(X_train.columns)).astype("str")
            X_train.to_parquet(f"{data_path}/{prep_folder}/part{i}.parquet", index=True)
        else:
            if i == 0:
                df = X_train.copy()
            else:
                df = df.append(X_train, ignore_index=True)
        del X_train
    
    if not parts:
        # save combined df
        df.columns = np.array(list(df.columns)).astype("str")
        df.to_parquet(f"{data_path}/{prep_folder}/prep.parquet")
        
    # save settings
    pd.Series(preprocess_args).to_csv(f"{data_path}/{prep_folder}/config.csv")

def merge_preprocessing_parts(data_path='Data/', prep_folder='prep/'):
    df = pd.concat([pd.read_parquet( f"{data_path}/{prep_folder}/part{i}.parquet" ) 
                    for i in range(4)], ignore_index=True)
    df.columns = np.array(list(df.columns)).astype("str")
    df.to_parquet(f"{data_path}/{prep_folder}/prep.parquet")


def preview_data_aug(datagen, X, y, IMG_SIZE=64):
    for X_batch, y_batch in datagen.flow(X, y, batch_size=12):
        f, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
        axes = np.ravel(axes)
        # create a grid of 4x3 images
        for i, ax in enumerate(axes):
            ax.imshow(X_batch[i].reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))
        # show the plot
        plt.show()
        break