#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 10:27:52 2020

@author: joeranbosma
"""


"""Create a unified set of functions which can handle the preprocessed
data as images, and accept the multi-output."""

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import wandb
from wandb.keras import WandbCallback

from starter_eda_model_funcs import get_model, MultiOutputDataGenerator, global_acc_lr_reduction_calback
from starter_eda_model_funcs import val_root_acc_lr_reduction_callback
from preprocessing import test_config, perform_preprocessing
from cross_validation_helper import cv_train_val_split
from helper import GlobalAccuracyCallback, generator_wrapper, to_one_hot
from helper import preview_data_aug, save_model

def generators_from_prep(datagen_args, preprocess_args, # settings
                         cross_val_num=0, cross_val_parts=8, # cross-validation settings
                         show_data_aug=False, batch_size=256, # other
                         train_or_test='train',
                         data_path='Data/', prep_path='Data/prep/'): # folders
    """Obtain train and validation generators from preprocessed images.
    
    Preprocessing can be done in advance, then the prep_path should contain
    the settings file 'config.pickle' with the same settings as provided to this function.
    If this is not the case, the preprocessing will be initiated by this function.
    """
    
    # set variables and load training labels
    image_width, image_height = preprocess_args['image_width'], preprocess_args['image_height']
    
    # check if preprocessing has been done, and coincides with current arguments
    success = test_config(preprocess_args, prep_path=prep_path)
    if success <= 1:
        print("Performing data preprocessing...")
        perform_preprocessing(preprocess_args, data_path=data_path, prep_path=prep_path, out='png')

    # read train labels
    train_df_ = pd.read_csv(f'{data_path}/{train_or_test}.csv')
    
    # add filename column to train labels df
    train_df_['filename'] = train_df_['image_id'] + '.png'
    
    # define data augmentation generator for multiple outputs
    train_datagen = MultiOutputDataGenerator(**datagen_args)
    val_datagen = MultiOutputDataGenerator({})
    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    # datagen.fit(x_train)
    
    # split the train and validation data
    # train_df, val_df = train_test_split(train_df_, test_size=0.08, random_state=576)
    if train_or_test == 'train':
        # convert target labels to one-hot encoding
        # this also returns the ordered labels of the newly created columns
        one_hot_columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
        train_df_, features = to_one_hot(train_df_, one_hot_columns=one_hot_columns)
        assert len(features) == 168 + 11 + 7, print(f"found {len(features)} one-hot encoded features")
        
        # split train and validation set
        train_df, val_df = cv_train_val_split(train_df_, cross_val_num=cross_val_num, 
                                              cross_val_parts=cross_val_parts, random_state=576)
    else:
        # test set
        test_df = train_df_
        test_generator = flow_from_prep(val_datagen, df=test_df, prep_path=prep_path, labels=[],
                                   image_size=(image_width, image_height), batch_size=batch_size,
                                   shuffle=False)
        return test_generator
    
    # couple the data generator to the prepared images
    train_generator = flow_from_prep(train_datagen, df=train_df, prep_path=prep_path, labels=features,
                                   image_size=(image_width, image_height), batch_size=batch_size)
    val_generator = flow_from_prep(val_datagen, df=val_df, prep_path=prep_path, labels=features,
                                   image_size=(image_width, image_height), batch_size=batch_size,
                                   shuffle=False)
    
    # Visualize few samples of current training dataset, including data augmentation
    if show_data_aug:
        preview_data_aug(train_generator)
        
    return train_generator, val_generator

def train(datagen_args, preprocess_args, name=None, batch_size=256, epochs=30, model=None, # settings
          cross_val_num=0, cross_val_parts=8, # cross-validation settings
          show_data_aug=True, # other
          data_path='Data/', prep_path='Data/prep/', model_path='Model/'): # folders
    """Train a model from preprocessed images.
        
        Preprocessing can be done in advance, then the prep_path should contain
        the settings file 'config.csv' with the same settings as provided to this function.
        If this is not the case, the preprocessing will be initiated by this function.
        """
    image_width, image_height = preprocess_args['image_width'], preprocess_args['image_height']
    
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
        
    # get train and validation generators
    train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args, 
                         cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, show_data_aug=show_data_aug,
                         batch_size=batch_size, data_path=data_path, prep_path=prep_path)
    
    # create custom global accuracy with weights 50%, 25%, 25%
    global_accuracy_callback = GlobalAccuracyCallback(validation_generator = val_generator)
    
    # get lr reduction on plateau callbacks
    lr_reduction = val_root_acc_lr_reduction_callback() #global_acc_lr_reduction_calback()
    
    # Fit the model, save every 10 epochs
    for it in range(epochs):
        # wrap the data generator to support multiple output labels
        _ = model.fit(generator_wrapper(train_generator), validation_data = generator_wrapper(val_generator),
                            initial_epoch = it, epochs = it+1, steps_per_epoch=train_generator.n//train_generator.batch_size,
                            validation_steps=val_generator.n//val_generator.batch_size,
                            callbacks=[lr_reduction, global_accuracy_callback, WandbCallback()])

        # save model offline every ten epochs, and after final epoch (if not a multitude of 10 already)
        if (((it+1) % 10) == 0) or ((it+1) == epochs):
            print("Saving model after {} epochs.".format(it+1))
            save_model(model, model_path=model_path, name=f"{name}-{it+1}-epochs")

    return model

def evaluate_trained_model(model, datagen_args, preprocess_args, # settings
                         cross_val_num=0, cross_val_parts=8, # cross-validation settings
                         name=None, show_data_aug=False, batch_size=256, # other
                         data_path='Data/', prep_path='Data/prep/'):
    """Evaluate performance of trained model"""
    
    # get train and validation generators
    train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args, 
                         cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, show_data_aug=show_data_aug,
                         batch_size=batch_size, data_path=data_path, prep_path=prep_path)
    
    # obtain metrics for validation set
    val_generator.reset()
    metrics = model.evaluate(generator_wrapper(val_generator), verbose=0,
                             steps=val_generator.n // val_generator.batch_size)

    # the metrics will contain out_root_acc, etc. for the individual accuracies
    metric_labels = model.metrics_names
    
    # calculate global accuracy
    worker = GlobalAccuracyCallback(val_generator)
    global_accuracy = worker.calc_global_acc(metrics, metric_labels)
    
    return global_accuracy, metrics, metric_labels

def preview_from_prep(datagen_args, preprocess_args, 
                      cross_val_num=0, cross_val_parts=8, # cross-validation settings
                      batch_size=256, nrows=3, ncols=4, # number of images (let batch_size > #num images)
                      data_path = 'Data/', prep_path = 'Data/prep/'):
    """Specifically preview data augmentation of preprocessed images"""

    # get train and validation generators
    train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args, 
                         cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, show_data_aug=False,
                         batch_size=batch_size, data_path=data_path, prep_path=prep_path)
    
    preview_data_aug(train_generator, nrows=nrows, ncols=ncols)



def flow_from_prep(datagen, df, prep_path, labels, image_size, batch_size, shuffle=True):
    return datagen.flow_from_dataframe(dataframe=df,
                                directory=prep_path,
                                x_col='filename',
                                y_col=labels,
                                class_mode='other',
                                target_size = image_size,
                                color_mode='grayscale',
                                batch_size=batch_size,
                                shuffle=shuffle)