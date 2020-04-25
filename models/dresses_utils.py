import cv2
import os
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, BatchNormalization, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np

color_names = ("beige","black","blue","brown","gray","green","multicolor","orange","pink","red","violet","white","yellow","transparent")

def read_number_of_lines(filename):
    with open(filename) as f:
        return len(f.read().splitlines())

def read_data(filename, number):
    if number == "*" :
        with open(filename) as f:
            lines = f.read().splitlines()
    else:
        with open(filename) as f:
            lines = f.read().splitlines()[:number]
    return lines

def create_images_labels(lines,directory):
    images = []
    labels = []

    for line in lines:
        tmp = line.split()
        path = tmp[0]
        label = tuple(map(int,tmp[1:]))
        path = os.path.join(directory, path)
        img = cv2.imread(path)
        images.append(img)
        labels.append(label)

    return images, labels

def rescale_image(img,size_tuple):
    return  cv2.resize(img,size_tuple,interpolation=cv2.INTER_AREA)

def create_model():
    model = Sequential()
    model.add(Conv2D(16,(3,3),input_shape = (128,128,3))) #warstwa wejściowa więc jest zdjęcie 128 128 x RGB (==3)
    model.add(BatchNormalization()) #normalizacja wyników w celu lepszego trenowania
    model.add(Activation("relu"))
    model.add(Conv2D(16,(3,3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation("relu"))
    model.add(Dense(14,activation = "sigmoid")) # because number of colours = 14
    return model
