#!/usr/bin/env python
import numpy as np
import cv2
import os
import sys
import math
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import rmsprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)

K.set_image_data_format('channels_first')

IMG_WIDTH = 80
IMG_HEIGHT = 60

def mask_edges(img):
    ignore_mask_color = 255
    mask = np.zeros_like(img)
    imshape = img.shape
    # vertices of a mask polygon
    lower_left = [0, imshape[0]]
    lower_right = [imshape[1], imshape[0]]
    top_left = [0, 0]
    top_right = [imshape[1], 0]
    top_middle = [imshape[1]/2, 0]
    center = [imshape[1]/2, imshape[0]/2]
    left_middle = [0, imshape[0]/2]
    right_middle = [imshape[1], imshape[0]/2]
    vertices = [np.array(
        [top_left, left_middle, right_middle, top_right],
        dtype=np.int32)]
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    copied_img = np.copy(img)
    pts = np.array([vertices], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(copied_img, [pts], True, (255, 255, 255))
    # cv2.imshow(?mask region?, copied_img)
    img = cv2.bitwise_and(img, mask)
    return img

def preprocess_img(img):
    #This function should be consistent with 'preprocess_img' in ml_driver.py
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #################
    #Task (optional): Any pre-processing?
    #################
    # mask
    img = mask_edges(img)
    # threshold
    _, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
    # flip
    img = cv2.flip(img, 1)

    return img

def load_dataset(image_path):
    owd = os.getcwd()
    filenames = sorted(os.listdir(image_path))
    os.chdir(image_path)
    num_files = len(filenames)
    X = []
    Y = []

    for i in range(num_files):
        fname = filenames[i]        
        angle = fname[:-4].split('_')[1]
        img = cv2.imread(fname)
        
        processed_img = preprocess_img(img)
        X.append(processed_img)
        Y.append(angle)
        
        #################
        #Task (optional): Data augmentation?
        #################


    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    X = X/255.0
    X = X.reshape(X.shape[0], 1, IMG_WIDTH, IMG_HEIGHT)

    os.chdir(owd)
    return X, Y

def create_model():
    model = Sequential()

    #################
    #Task: Create your own model
    #################
    model.add(Conv2D(32, (3, 2), padding='same',
                input_shape=(1, IMG_WIDTH, IMG_HEIGHT),
                activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same',
                activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same',
                activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same',
                activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('USAGE: ./train.py model_file_name')
        sys.exit(1)
    
    model_file_name = sys.argv[1]

    model = create_model()    
    
    X1, Y1 = load_dataset('set1')
    X2, Y2 = load_dataset('set2')
    X3, Y3 = load_dataset('set3')
    X4, Y4 = load_dataset('set4')
        
    X_train = np.concatenate((X2, X3, X4))
    Y_train = np.concatenate((Y2, Y3, Y4))
    
    X_test = X1[2000:]
    Y_test = Y1[2000:]

    #################
    #Task (optional): You can tune these parameters
    batch_size = 32
    epochs = 30
    #################

    model.compile(loss='mse', optimizer='adam')

    history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=[X_test, Y_test],
            callbacks=[ModelCheckpoint(model_file_name)]
            )

    print("--------------------------------------------------------------------------------")
    print("Training is completed. Copy %s to your car using scp command." % model_file_name)
