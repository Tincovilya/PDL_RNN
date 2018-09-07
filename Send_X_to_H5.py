# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:18:14 2018

@author: Nic Smith

This module literally just takes the X's from Build_Examples and turns them
into an h5 array, because that makes everything way more effecient.
"""

import pandas as pd
import numpy as np
import Build_Examples
from tqdm import tqdm

def Send_X_To_h5():
    #First get the times from the AGM list
    times = Build_Examples.insert_examples()

    #Next build all of the examples (this takes about 20 minutes)
    X=[]
    for j in tqdm(times):
        X.append(Build_Examples.get_xs(j[0],j[1]))

    #Write information into a dataframe
    df = pd.DataFrame.from_records(X)

    print(df)

    store = pd.HDFStore("X_Examples.h5")
    store.append("X", df)
