# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:03:16 2018

@author: nsmith

This module exists only to get the X examples BACK from an h5 and then
toss that into the main as a numpy array
"""
import pandas as pd


def get_X():
    store = pd.HDFStore("X_Examples.h5")
    df_X = store["X"]
    
    return df_X.values
