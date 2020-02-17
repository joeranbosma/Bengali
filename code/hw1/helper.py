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
import time

import tensorflow as tf
from matplotlib import pyplot as plt


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
    
    def calc_global_acc(self, metrics, metric_labels):
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
    
    def eval_acc(self):
        # obtain metrics for validation set
        self.val_gen.reset()
        metrics = self.model.evaluate(generator_wrapper(self.val_gen), verbose=0,
                                      steps=self.val_gen.n // self.val_gen.batch_size)
        # the metrics will contain out_root_acc, etc. for the individual accuracies
        metric_labels = self.model.metrics_names
        
        # calculate global accuracy
        global_acc = self.calc_global_acc(metrics, metric_labels)

        return global_acc

    def on_epoch_end(self, epoch, logs={}):
        acc = self.eval_acc()
        # print("Global accuracy for epoch %d is %f"%(epoch, acc))
        self.accs.append(acc)
        wandb.log({'val_global_accuracy': acc}, step=wandb.run.step)

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

def save_model(model, model_path, name):
    try:
        path = "{}/{}/".format(model_path, name)
        if not os.path.exists(path): os.makedirs(path)
        model.save("{}/model-trained-{}.h5".format(path, time.strftime("%Y%m%d-%H%M%S")))
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
