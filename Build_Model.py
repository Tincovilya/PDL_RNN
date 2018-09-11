# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:19:05 2018

@author: nicpa

Creates an RNN of depth 3 with all CuDNNGRU units. The results will be either
a 1 or a 0 telling the probable class of EVERY millisecond of input data.

Uses Keras.

Note that DR = dropout rate
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, TimeDistributed

def create_model(x_train, GRU_units,DR):
    
    model = Sequential()
    print("Adding first layer...")
    model.add(CuDNNLSTM(GRU_units,
                      input_shape = (x_train.shape[1:]),
                      return_sequences=True))
    model.add(Dropout(DR))
    print(model.output_shape)
    
    print("Adding second layer...")
    model.add(CuDNNLSTM(GRU_units,
                       return_sequences=True))
    model.add(Dropout(DR))
    print(model.output_shape)
    
    print("Adding third layer...")
    model.add(CuDNNLSTM(GRU_units, 
                        return_sequences=True))
    model.add(Dropout(DR))
    print(model.output_shape)    
    
    print("Adding fourth layer (Dense 32)...")
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DR))
    print(model.output_shape)
    
    print("Adding output layer...")
    model.add(Dense(2, activation='softmax'))
    print(model.output_shape)
    
    return model

