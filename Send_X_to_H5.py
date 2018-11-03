# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:18:14 2018

@author: Nic Smith

This module literally just takes the X's from Build_Examples and turns them
into an h5 array, because that makes everything way more effecient.
"""

import h5py
import Build_Examples
from tqdm import tqdm

def Send_X_To_h5(times,maxi):
    
    hf = h5py.File("X_Examples.h5", 'w')
    #Next build all of the examples (this takes about 20 minutes)
    for i, j in enumerate(tqdm(times)):
        hf.create_dataset(str(i), data = Build_Examples.get_xs(j[0],j[1],maxi))
    hf.close()
