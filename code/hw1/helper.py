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
from starter_eda_model_funcs import get_lr_reduction_calbacks, global_acc_lr_reduction_calback
from preprocessing import perform_preprocessing, test_config
from cross_validation_helper import cv_train_val_split


# adapted from https://github.com/keras-team/keras/issues/4506
class GlobalAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_generator):
        self.val_gen = validation_generator
        self.accs = []

    def eval_acc(self):
        # obtain metrics for validation set
        self.val_gen.reset()
        metrics = self.model.evaluate(generator_wrapper(self.val_gen), verbose=0,
                                      steps=self.val_gen.n // self.val_gen.batch_size)
        # the metrics will contain out_root_acc, etc. for the individual accuracies
        metric_labels = self.model.metrics_names
        
        # calculate global accuracy
        global_acc = 0
        # define weights to each part
        global_weights = {'out_root_acc': 0.5, 'out_vowel_acc': 0.25, 'out_consonant_acc': 0.25}
        assert np.sum(list(global_weights.values())) == 1., "Set weights to sum to one, for normalization"
        for key, weight in global_weights.items():
            # search for the index of the accuracy (root/vowel or consonant) in the metric list
            idx = metric_labels.index(key)
            global_acc += metrics[idx] * weight

        return global_acc

    def on_epoch_end(self, epoch, logs={}):
        acc = self.eval_acc()
        # print("Global accuracy for epoch %d is %f"%(epoch, acc))
        self.accs.append(acc)
        wandb.log({'val_global_accuracy': acc}, step=wandb.run.step)

# adapted from https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn
# added WandB integration and custom datagen arguments
def train_from_prep(datagen_args, name=None, epochs=30, model=None,
                    N_CHANNELS=1, batch_size=256, preprocess_args={'image_width': 64, 'image_height': 64, 'padding': 6},
          data_path='Data/', prep_folder='prep/', model_path='Model/',
                   cross_val_parts=8, cross_val_num=0):
    """Train a model from preprocessed images.
        
        Preprocessing can be done in advance, then the prep_folder should contain
        the settings file 'config.csv' with the same settings as provided to this function.
        If this is not the case, the preprocessing will be initiated by this function.
        """
    
    # set variables and load training labels
    image_width, image_height = preprocess_args['image_width'], preprocess_args['image_height']
    prep_path = data_path + prep_folder

    # check if preprocessing has been done, and coincides with current arguments
    success = test_config(preprocess_args, prep_path=prep_path)
    if success <= 1:
        print("Performing data preprocessing...")
        perform_preprocessing(preprocess_args, data_path=data_path, prep_folder=prep_folder, out='png')

    # read train labels
    train_df_ = pd.read_csv(f'{data_path}/train.csv')
    
    # set up config
    config = datagen_args.copy()
    config['epochs'] = epochs
    config['name'] = name
    config.update(preprocess_args)
    if image_width == image_height:
        config['image_size'] = image_width
    wandb.init(project='mlip', name=name, config=config)
    
    if model == None:
        assert image_width == image_height, "function get_model not yet ready for rectanglurar images"
        model = get_model(img_size=image_width)
    
    # add filename column to train labels df
    train_df_['filename'] = train_df_['image_id'] + '.png'

    # convert target labels to one-hot encoding
    # this also returns the ordered labels of the newly created columns
    train_df_, features = to_one_hot(train_df_, one_hot_columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])
    assert len(features) == 168 + 11 + 7, print(f"found {len(features)} one-hot encoded features")

    # define data augmentation generator for multiple outputs
    train_datagen = MultiOutputDataGenerator(**datagen_args)
    val_datagen = MultiOutputDataGenerator({})
    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    # datagen.fit(x_train)
    
    # split the train and validation data
    # train_df, val_df = train_test_split(train_df_, test_size=0.08, random_state=576)
    train_df, val_df = cv_train_val_split(train_df_, cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, random_state=576)

    # couple the data generator to the prepared images
    train_generator = flow_from_prep(train_datagen, df=train_df, prep_path=prep_path, labels=features,
                                   image_size=(image_width, image_height), batch_size=batch_size)
    val_generator = flow_from_prep(val_datagen, df=val_df, prep_path=prep_path, labels=features,
                                   image_size=(image_width, image_height), batch_size=batch_size)
    
    # Visualize few samples of current training dataset, including data augmentation
    preview_data_aug(train_generator)

    # create custom global accuracy with weights 50%, 25%, 25%
    global_accuracy_callback = GlobalAccuracyCallback(validation_generator = val_generator)

    # get lr reduction on plateau callbacks
    lr_reduction = global_acc_lr_reduction_calback()

    # Fit the model
    # wrap the data generator to support multiple output labels
    history = model.fit(generator_wrapper(train_generator), validation_data = generator_wrapper(val_generator),
                        epochs = epochs, steps_per_epoch=train_generator.n//train_generator.batch_size,
                        validation_steps=val_generator.n//val_generator.batch_size,
                        callbacks=[lr_reduction, global_accuracy_callback, WandbCallback()])
    # save model offline
    save_model(model, model_path=model_path, name=name)
    
    return model

def flow_from_prep(datagen, df, prep_path, labels, image_size, batch_size):
    return datagen.flow_from_dataframe(dataframe=df,
                                directory=prep_path,
                                x_col='filename',
                                y_col=labels,
                                class_mode='other',
                                target_size = image_size,
                                color_mode='grayscale',
                                batch_size=batch_size )

def generator_wrapper(generator):
    labels = ['out_root', 'out_vowel', 'out_consonant']
    lengths = [168,         11,          7]
    # create start and end indices (0:168, 168:168+11, ...)
    stop = list(np.cumsum(lengths))
    start = [0]
    start.extend(stop)
    # sth. with cumsum to improve hard coded below?
    for batch_x,batch_y in generator:
        # print("Sum ", np.sum(batch_y))
        yield (batch_x, {labels[i]: batch_y[:, start[i]:stop[i]] for i in range(3)})

def to_one_hot(df, one_hot_columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']):
    # Convert categorical column(s) to one-hot encodings
    features = []

    for col in one_hot_columns:
        if col in df.columns:
            print("Converting {} to one-hot encoding".format(col))
            onehot = pd.get_dummies(df[col], prefix=col)
            features.extend(list(onehot.columns))
            df = pd.merge(onehot, df, left_index=True, right_index=True)
            df.drop(col, axis=1, inplace=True)

    return df, features

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

def save_model(model, model_path, name):
    try:
        model.save("{}/{}/model-trained-{}.h5".format(model_path, name, time.strftime("%Y%m%d-%H%M%S")))
    except:
        print("Model save failed, retrying as model.h5 in current directory")
        try:
            model.save("model.h5")
        except:
            print("Model save failed again.")
    
    # save model online
    try:
        model.save(os.path.join(wandb.run.dir, "model-trained.h5"))
    except:
        pass

def preview_data_aug(img_generator, nrows=3, ncols=4):
    f, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    x_batch, y_batch = img_generator.next()
    for i, (ax, x) in enumerate(zip(np.ravel(axes), x_batch)):
        ax.imshow(x.squeeze(), cmap='gray')
        ax.set_axis_off()
    plt.show()

def preview_from_prep(preprocess_args, datagen_args, data_path = 'Data/', prep_folder = 'prep/', 
                      batch_size=256, nrows=3, ncols=4):
    """Basically train from prep, without model and training"""

    # set variables and load training labels
    prep_path = data_path + prep_folder
    print(f"Flowing from {prep_path}")

    image_width, image_height = preprocess_args['image_width'], preprocess_args['image_height']

    # read train labels
    train_df_ = pd.read_csv(f'{data_path}/train.csv')

    # add filename column to train labels df
    train_df_['filename'] = train_df_['image_id'] + '.png'

    # convert target labels to one-hot encoding
    # this also returns the ordered labels of the newly created columns
    train_df_, features = to_one_hot(train_df_, one_hot_columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])
    assert len(features) == 168 + 11 + 7, print(f"found {len(features)} one-hot encoded features")

    # define data augmentation generator for multiple outputs
    train_datagen = MultiOutputDataGenerator(**datagen_args)
    val_datagen = MultiOutputDataGenerator({})

    # split the train and validation data
    # train_df, val_df = train_test_split(train_df_, test_size=0.08, random_state=666)
    train_df, val_df = cv_train_val_split(train_df_, cross_val_num=0, cross_val_parts=8, random_state=576)

    # couple the data generator to the prepared images
    train_generator = flow_from_prep(train_datagen, df=train_df, prep_path=prep_path, labels=features,
                                    image_size=(image_width, image_height), batch_size=batch_size)
    
    preview_data_aug(train_generator, nrows=nrows, ncols=ncols)
