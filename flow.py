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
import gc

from tensorflow.keras.callbacks import LearningRateScheduler
import wandb
from wandb.keras import WandbCallback

from starter_eda_model_funcs import get_model, MultiOutputDataGenerator, global_acc_lr_reduction_calback
from starter_eda_model_funcs import val_root_acc_lr_reduction_callback
from preprocessing import test_config, perform_preprocessing
from preprocessing import crop_resize
from cross_validation_helper import cv_train_val_split
from helper import GlobalAccuracyCallback, generator_wrapper, to_one_hot
from helper import preview_data_aug, save_model


def generators_from_prep(datagen_args, preprocess_args,               # settings
                         cross_val_num=0, cross_val_parts=8,          # cross-validation settings
                         show_data_aug=False, batch_size=256,         # other
                         train_or_test='train',
                         data_path='Data/', prep_path='Data/prep/'):  # folders
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
    train_df_ = pd.read_csv('{}/{}.csv'.format(data_path, train_or_test))

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
        assert len(features) == 168 + 11 + 7, print("found {} one-hot encoded features".format(len(features)))

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


# Test submission code from https://www.kaggle.com/ipythonx/keras-grapheme-gridmask-augmix-in-efficientnet
# Test data generator with preprocessing on the fly
def test_batch_generator(df, batch_size, SIZE, PAD, HEIGHT=137, WIDTH=236):
    num_imgs = len(df)

    for batch_start in range(0, num_imgs, batch_size):
        curr_batch_size = min(num_imgs, batch_start + batch_size) - batch_start
        idx = np.arange(batch_start, batch_start + curr_batch_size)

        names_batch = df.iloc[idx, 0].values
        imgs_batch = 255 - df.iloc[idx, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        X_batch = np.zeros((curr_batch_size, SIZE, SIZE, 1))

        # perform preprocessing on the fly
        for j in range(curr_batch_size):
            img = (imgs_batch[j,]*(255.0/imgs_batch[j,].max())).astype(np.uint8)
            img = crop_resize(img, orig_height=HEIGHT, orig_width=WIDTH, target_height=SIZE, target_width=SIZE, pad=PAD)
            img = img[:, :, np.newaxis]
            X_batch[j,] = img

        yield X_batch, names_batch


def predict_with_prep_on_the_fly(model, preprocess_args, data_path='Data/', train_or_test='test',
                                 batch_size=256):
    file_names = ["{}/{}_image_data_{}.parquet".format(data_path, train_or_test, i)
                  for i in range(4)]

    # placeholders
    row_id = []
    target = []
    probs = {}

    # iterative over the test sets
    for fname in tqdm(file_names):
        test_ = pd.read_parquet(fname)
        assert preprocess_args['image_width'] == preprocess_args['image_height'], "rect. images not implemented"
        test_gen = test_batch_generator(test_, batch_size=batch_size, SIZE=preprocess_args['image_width'],
                                        PAD=preprocess_args['padding'])

        for batch_x, batch_name in test_gen:
            batch_predict = model.predict(batch_x)
            for idx, name in enumerate(batch_name):
                # save probabilities
                probs[f"{name}_consonant_diacritic"] = batch_predict[2][idx]
                probs[f"{name}_grapheme_root"] = batch_predict[0][idx]
                probs[f"{name}_vowel_diacritic"] = batch_predict[1][idx]

        del test_
        gc.collect()

    return probs


def evaluate_trained_model(model, datagen_args, preprocess_args, # settings
                           cross_val_num=0, cross_val_parts=8, # cross-validation settings
                           name=None, show_data_aug=False, batch_size=256, # other
                           data_path='Data/', prep_path='Data/prep/'):
    """Evaluate performance of trained model"""

    if prep_path is not None:
        # get train and validation generators
        train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args,
                             cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, show_data_aug=show_data_aug,
                             batch_size=batch_size, data_path=data_path, prep_path=prep_path)

    # obtain metrics for validation set
    val_generator.reset()
    metrics = model.evaluate(generator_wrapper(val_generator), verbose=1,
                             steps=val_generator.n // val_generator.batch_size)

    # the metrics will contain out_root_acc, etc. for the individual accuracies
    metric_labels = model.metrics_names

    # calculate global accuracy
    worker = GlobalAccuracyCallback(val_generator)
    global_accuracy = worker.calc_global_acc(metrics, metric_labels)

    return global_accuracy, metrics, metric_labels


def train(datagen_args, preprocess_args, name=None, batch_size=256, epochs=30, model=None,  # settings
          cross_val_num=0, cross_val_parts=8, lr_scheduler_func=None,  # cross-validation settings
          show_data_aug=True,  # other
          webdav_client=None, min_epoch_upload=10, external_path='models/',# upload models to webdav client
          data_path='Data/', prep_path='Data/prep/', model_path='Model/'):  # folders
    """Train a model from preprocessed images.

        Preprocessing can be done in advance, then the prep_path should contain
        the settings file 'config.csv' with the same settings as provided to this function.
        If this is not the case, the preprocessing will be initiated by this function.
        """
    image_width, image_height = preprocess_args['image_width'], preprocess_args['image_height']

    if model == None:
        assert image_width == image_height, "function get_model not yet ready for rectanglurar images"
        model = get_model(img_size=image_width)

    # get train and validation generators
    train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args,
                                                          cross_val_num=cross_val_num, cross_val_parts=cross_val_parts,
                                                          show_data_aug=show_data_aug,
                                                          batch_size=batch_size, data_path=data_path,
                                                          prep_path=prep_path)

    # create custom global accuracy with weights 50%, 25%, 25%
    global_accuracy_callback = GlobalAccuracyCallback(validation_generator=val_generator)

    # define callbacks
    callbacks = [global_accuracy_callback]
    if lr_scheduler_func is not None:
        callbacks.append(LearningRateScheduler(lr_scheduler_func, verbose=1))

    # get lr reduction on plateau callbacks
    # lr_reduction = global_acc_lr_reduction_calback()

    # create folder in webdav client
    # if webdav_client is not None:
    #     webdav_client.execute_request("mkdir", '/{}/{}/'.format(model_path, name))

    # set up config and start Weights & Biases run
    config = datagen_args.copy()
    config['epochs'] = epochs
    config['name'] = name
    config.update(preprocess_args)
    if image_width == image_height:
        config['image_size'] = image_width
    wandb.init(project='mlip', name=name, config=config)

    callbacks.append( WandbCallback(monitor='val_global_accuracy', verbose=1, mode='max',
                                    log_best_prefix='best_') )

    # Fit the model, save every 10 epochs and if global accuracy improved
    val_global_accuracy_best = 0
    for ep in range(1, 1 + epochs):
        # wrap the data generator to support multiple output labels
        _ = model.fit(generator_wrapper(train_generator), validation_data=generator_wrapper(val_generator),
                      initial_epoch=ep - 1, epochs=ep, steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=val_generator.n // val_generator.batch_size,
                      callbacks=callbacks)

        # check global validation accuracy
        val_glob_acc = wandb.run.summary["val_global_accuracy"]
        if val_glob_acc > val_global_accuracy_best and ep >= min_epoch_upload:
            val_global_accuracy_best = val_glob_acc
            model_fn = "model-best.h5"
            print("Saving new best model, with val global accuracy of {:.6f} to {}".format(val_glob_acc, model_fn))
            model.save(model_fn)
            if webdav_client is not None:
                print("Uploading async...")
                # Unload resource
                kwargs = {
                    'remote_path': "{}/model-best-{ep:03d}.h5".format(external_path, ep=ep),
                    'local_path': model_fn,
                    'callback': lambda: print("Upload finished.")
                }
                webdav_client.upload_async(**kwargs)

    return model


def evaluate_trained_model(model, datagen_args, preprocess_args, # settings
                           cross_val_num=0, cross_val_parts=8, # cross-validation settings
                           name=None, show_data_aug=False, batch_size=256, # other
                           data_path='Data/', prep_path='Data/prep/'):
    """Evaluate performance of trained model"""

    if prep_path is not None:
        # get train and validation generators
        train_generator, val_generator = generators_from_prep(datagen_args=datagen_args, preprocess_args=preprocess_args,
                             cross_val_num=cross_val_num, cross_val_parts=cross_val_parts, show_data_aug=show_data_aug,
                             batch_size=batch_size, data_path=data_path, prep_path=prep_path)

    # obtain metrics for validation set
    val_generator.reset()
    metrics = model.evaluate(generator_wrapper(val_generator), verbose=1,
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


def get_lr_test_scheduler(lr_start, lr_end, num):
    def lr_scheduler(epoch, lr):
        lr_range = np.linspace(lr_start, lr_end, num=num)
        lr = lr_range[epoch]
        return lr
    return lr_scheduler


def get_pyramid_lr(lr_start=0.01, lr_max=0.5, n_epoch=100, n_epochs_end=10):
    def lr_pyramid(epoch, lr):
        n_epoch_half_pyramid = int(np.ceil(n_epoch/2-n_epochs_end/2))
        lr_range = np.concatenate((
            np.linspace(lr_start, lr_max, num=n_epoch_half_pyramid),
            np.linspace(lr_max, lr_start, num=n_epoch_half_pyramid),
            np.linspace(lr_start, lr_start/10, num=n_epochs_end)))
        lr = lr_range[epoch]
        return lr
    return lr_pyramid
