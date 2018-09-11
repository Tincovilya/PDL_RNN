# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:03:16 2018

@author: nsmith

This module exists only to get the X examples BACK from an h5 and then
toss that into the main as a numpy array
"""
import numpy as np
import h5py

def get_X(times):
    X_temp = []
    hf = h5py.File("X_Examples.h5", 'r')
    for i,j in enumerate(times):
        x = hf.get(str(i))
        x = np.array(x)
        X_temp.append(x)
    
    hf.close()
    
    return X_temp
