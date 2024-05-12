import os
import keras
from keras import callbacks, Model
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, add

from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime

def listdir_nohidden(path):
    lst = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            if f.endswith('.tif'):
                lst.append(f)
    lst = sorted(lst)
    return lst

def preparing_data(folder):
    """
    Preparing data for training
    create dataset from folder with tif files (DEMs)
    """
    # Загрузка данных
    #dataset shapes
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1
    X_train_ids = listdir_nohidden (folder)
    print ("Count in high_res_folder: " + str(len(X_train_ids)))
    X_train = np.zeros((len(X_train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    for n, id_ in tqdm(enumerate(X_train_ids), total=len(X_train_ids)):
        path = folder
        if id_.endswith(".tif"):
            img = imread(path + '/' + id_)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_train[n] = img
    
    X_train_reshape = X_train.astype('float32') / 10000.0
    print (X_train_reshape.shape)
    
    return X_train_reshape


def preparing_largedata(folder):
    IMG_HEIGHT = 1940
    IMG_WIDTH = 1964
    IMG_CHANNELS = 1
    """
    Preparing data for training
    create dataset from folder with tif files (DEMs)
    """
    X_train_ids = listdir_nohidden (folder)
    print ("Count in high_res_folder: " + str(len(X_train_ids)))
    X_train = np.zeros((len(X_train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    for n, id_ in tqdm(enumerate(X_train_ids), total=len(X_train_ids)):
        path = folder
        if id_.endswith(".tif"):
            img = imread(path + '/' + id_)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_train[n] = img
    
    X_train_reshape = X_train.astype('float32') / 10000.0
    print (X_train_reshape.shape)
    
    return X_train_reshape

def prepare_and_augment_data(folder):
    """
    Preparing and augmenting data for training
    Create and augment dataset from folder with tif files (DEMs)
    """
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    IMG_CHANNELS = 1
    X_train_ids = listdir_nohidden(folder)
    print("Count in folder: " + str(len(X_train_ids)))
    # Multiply by 8 due to augmentation (original, 3 rotations, and 4 mirrored versions)
    X_train = np.zeros((len(X_train_ids) * 8, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    
    for n, id_ in tqdm(enumerate(X_train_ids), total=len(X_train_ids)):
        path = folder
        img_path = os.path.join(path, id_)
        img = imread(img_path)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = img.astype('float32') / 10000.0  # Normalize data

        # Original image
        X_train[n * 8] = np.expand_dims(img, axis=-1)
        # Rotations and mirror flips
        for k in range(1, 4):  # Add rotated images
            X_train[n * 8 + k] = np.expand_dims(np.rot90(img, k), axis=-1)
        img_flip_lr = np.fliplr(img)
        img_flip_ud = np.flipud(img)
        img_flip_both = np.flipud(np.fliplr(img))
        X_train[n * 8 + 4] = np.expand_dims(img_flip_lr, axis=-1)
        X_train[n * 8 + 5] = np.expand_dims(img_flip_ud, axis=-1)
        X_train[n * 8 + 6] = np.expand_dims(img_flip_both, axis=-1)
        X_train[n * 8 + 7] = np.expand_dims(np.rot90(img_flip_both), axis=-1)  # Rotated mirrored

    print(X_train.shape)
    
    return X_train


if __name__ == "__main__":
    print ("")