# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:10:04 2018

@author: nicpa

This will return a full set of test and training data.

Requires:
    1) An X_Examples.h5
    2) The "Times" that you can build samples around
    
"""
import numpy as np
import random
from tqdm import tqdm
import Build_Examples
import Retrieve_X_H5
from sklearn import preprocessing

def setup(times):
    #Build y array
    Y=[]
    for j in tqdm(times):
        Y.append(Build_Examples.insert_ones(np.zeros((5000,1)), j[0], j[2], j[3],5000))
    
    #Read in X array
    X = Retrieve_X_H5.get_X()
    
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
        
    return (train_X, train_Y, test_X, test_Y)