# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:21:07 2018

@author: nsmith
"""

import numpy as np
import random

from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from keras.layers import GRU, BatchNormalization
from keras.optimizers import Adam
from sklearn import preprocessing

Tx = 5000 #Number of data points in a sample
n_features = 7 #Number of columns (features) in each data sample

def setup(X,Y,Ty):
    items = range(len(X))
    sample = set(random.sample(items, 312)) #Get a sample of which indicies for train
    train_mask = sorted(sample)
    test_mask = sorted(set(items)-sample)
    #Normalize data
    normalized_X = []
    for i in X:
        normalized_X.append(preprocessing.normalize(i, copy=False))
    
    #Create training set from the indexes made above
    train_X = []
    train_Y = []
    for i in train_mask:
        train_X.append(normalized_X[i])
        train_Y.append(Y[i])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    
    #Create testing set from the indexes made above
    test_X = []
    test_Y = []
    for i in test_mask:
        test_X.append(normalized_X[i])
        test_Y.append(Y[i])
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
        
    return (train_X, train_Y, test_X, test_Y, train_mask,test_mask)

def model_func(input_shape, num_filters, kernel_size_passed, GRU_units,stride):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    #CONV layer
    X = Conv1D(filters=num_filters, kernel_size=kernel_size_passed, strides=stride)(X_input)   # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                   # ReLu activation
    X = Dropout(0.8)(X)                                         # dropout (use 0.8)

    #First GRU Layer
    X = GRU(units = GRU_units, return_sequences = True)(X)    # GRU (return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                         # Batch normalization
    
    #Second GRU Layer
    X = GRU(units = GRU_units, return_sequences = True)(X)    # GRU (return the sequences)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                         # Batch normalization
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    
    #Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

def main(X,Y,Ty,num_filters,kernel_size,GRU_units,stride):
    #Get the training and testing sets
    train_X, train_Y, test_X, test_Y, train_mask, test_mask = setup(X,Y,Ty)
    #Get the model
    input_shape = (Tx, n_features)
    model = model_func(input_shape,num_filters,kernel_size,GRU_units,stride)
    
    #Tell me what that looks like
    model.summary()
    
    #Fit model
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    model.fit(train_X, train_Y, batch_size = 10, epochs=30)
    
    return model, train_mask, test_mask, test_X, test_Y