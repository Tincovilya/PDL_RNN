# -*- coding: utf-8 -*-
"""
Auth: Nicholas Smith
Date: March 5, 2018
Version: 2

This script will take acceleration (X,Y,Z) data from a smart guage and make some
visualizations and calculations that will show velocity, and help to see where
there were incidents. No optimizing
"""
import pandas as pd
import datetime as dt
import PreProcess_Data
import Graphing
from tqdm import tqdm
import os
import numpy as np

"""
Going to first do velocity = a sum of all incidents of Z acceleration over time
keeping in mind the sampling rates are different for each second interval
so I will also have to find out the sampling rate then divide each temporary
velocity sum by that number.
This is an example of how to look at specific intervals of time
df_test = df_acc[df_acc.Time.dt.strftime('%H:%M:%S.%f').between('15:06:25.560','15:07:48.869')]
"""
PreProcess_Data.main(cur_path,"Temp")
PreProcess_Data.main(cur_path,"Gyro")

cur_path = os.path.dirname(os.path.abspath(__file__))
store=pd.HDFStore("Sensor_Data.h5")

nrows = store.get_storer("Temp").nrows
press = np.empty([0,3])
curr_low=pd.read_hdf(store,"Temp",where="index=0")["Pressure"][0]
curr_high=0.0
if nrows > 500000:
    prev = 0
    for i in range(0,nrows,500000):
        df_press = pd.read_hdf(store,"Temp",where="index<="+str(i)+" and index>"+str(prev))
        if len(df_press) > 0:
            ind = prev + 1
            prev_sec = df_press.Time[ind].second
            for j in df_press.Time:
                if j.second > prev_sec:
                    press = np.append(press,[[j,curr_low,curr_high]], axis=0)
                    prev_sec = j.second
                    curr_low=curr_high
                    curr_high=0
                if df_press.Pressure[ind] > curr_high:
                    curr_high = df_press.Pressure[ind]
                elif df_press.Pressure[ind] < curr_low:
                    curr_low = df_press.Pressure[ind]
                ind+=1
                if prev_sec == 59:
                    prev_sec = -1

        prev = i
else:
    df_press = pd.read_hdf(store,"Temp")
    ind = 0
    prev_sec = 0
    for j in df_press.Time:
        if j.second > prev_sec:
            press = np.append(press,[[j,curr_low,curr_high]])
            prev_sec = j.second
            curr_low=0
            curr_high=0
        if df_press.Pressure[ind] > curr_high:
            curr_high = df_press.Pressure[ind]
        elif df_press.Pressure[ind] < curr_low:
            curr_low = df_press.Pressure[ind]
        ind+=1
        if prev_sec == 59:
            prev_sec = -1

acc_combine = []
vel_combine = []
dist_combine = []
temp_vel = 0
temp_dist = 0

title = "Prince Albert Loop - Preliminary Examination"
df_AGM_List = pd.read_csv(cur_path +"\\AGMs.csv",index_col=None, header=0)
df_AGM_List.columns = ["AGM","Chainage","Elevation","Day","Time","Type"]
Graphing.main(df_press,vel_combine,df_AGM_List,title)
