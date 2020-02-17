# adapted from https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn

import pandas as pd
from tqdm import tqdm_notebook as tqdm
import cv2
import numpy as np

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


def resize(df, resize_size=64, need_progress_bar=True):
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

        roi = image[ymin:ymax,xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
        resized[df.index[i]] = resized_roi.reshape(-1)
    
    resized = pd.DataFrame(resized).T
    return resized

def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)

def get_model(img_size = 64, loss_weights=[0.5, 0.25, 0.25]):
    inputs = Input(shape = (img_size, img_size, 1))

    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(inputs)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
    model = BatchNormalization(momentum=0.15)(model)
    model = Dropout(rate=0.3)(model)

    model = Flatten()(model)
    model = Dense(1024, activation = "relu")(model)
    model = Dropout(rate=0.3)(model)
    dense = Dense(512, activation = "relu")(model)

    head_root = Dense(168, activation = 'softmax', name='out_root')(dense)
    head_vowel = Dense(11, activation = 'softmax', name='out_vowel')(dense)
    head_consonant = Dense(7, activation = 'softmax', name='out_consonant')(dense)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'],
                  loss_weights=loss_weights)

    return model

class MultiOutputDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict

def global_acc_lr_reduction_calback(patience=3, verbose=1, factor=0.5, min_lr=0.00001):
    # Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_global_accuracy', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    return learning_rate_reduction

def val_root_acc_lr_reduction_callback(patience=3, verbose=1, factor=0.5, min_lr=0.00001):
    # Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_out_root_acc', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    return learning_rate_reduction

def get_lr_reduction_calbacks(patience=3, verbose=1, factor=0.5, min_lr=0.00001):
    # Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
    learning_rate_reduction_root = ReduceLROnPlateau(monitor='out_root_acc', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='out_vowel_acc', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='out_consonant_acc', 
                                                patience=3, 
                                                verbose=1,
                                                factor=0.5, 
                                                min_lr=0.00001)
    return learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant
