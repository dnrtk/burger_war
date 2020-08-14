# -*- coding: utf-8 -*-

import os
import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input, Dropout, GlobalAveragePooling2D, Concatenate
from keras.layers.core import Dense, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPool2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import rl
import numpy as np

def createInceptionv3Module(input, outputFilter):
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell1)
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell1)

    cell2 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)
    cell2 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell2)

    cell3 = MaxPooling2D(pool_size=2, strides=1, padding='same')(input)
    cell3 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(cell3)

    cell4 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)

    output = Concatenate()([cell1, cell2, cell3, cell4])
    return output

class MyModel:
    def __init__(self, modelPath='./MyModel.h5'):
        if os.path.isfile(modelPath):
            self.model = load_model(modelPath)
            print('Load MyModel : {}'.format(modelPath))
        else:
            self.model = self.createModel()
            print('Create New MyModel')
    
    def createModel(self):
        input = Input(shape=(80, 80, 6))
        x = createInceptionv3Module(input, 32)
        x = createInceptionv3Module(x, 64)
        x = createInceptionv3Module(x, 128)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(3, activation='tanh')(x)
        model = Model(input, output)
        model.summary()

        return model

    def createModelCNN(self):
        input = Input(shape=(80, 80, 6))
        x = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(input)
        x = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(3, activation='tanh')(x)
        model = Model(input, output)
        model.summary()

        return model

    def predict(self, envImg):
        predictResult = self.model.predict(envImg)
        # print(predictResult)
        tempX = predictResult[0][0] * 1.4
        tempY = predictResult[0][1] * 1.4
        tempR = predictResult[0][2] * 180.0
        return (tempX, tempY, tempR)

