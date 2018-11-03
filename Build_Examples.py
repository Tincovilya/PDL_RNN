# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:51:54 2018

This file is going to create the examples "X" that will be fed into my RNN for training and
development purposes.

The examples will all be 3 second clips of Acceleration, Gyro, and pressure data with
the welds being not centered. So initially it will read in a list of time stamps
and then it will use these to build the samples to feed into the RNN with labelling
so it will also generate Y

@author: nsmith
"""

import pandas as pd
import numpy as np
import datetime as dt

def insert_examples():
    weld_times = pd.read_csv("RNN Data Times.txt",sep='\t', header=0,names=["id","Date","Start","End","Regime"],
                             dtype={"id":"int64","Date":"str","Start":"str","End":"str","Regime":"float64"})
    dates = pd.Series(weld_times.Date, index=weld_times.index)
    weld_times["Start_Change"] = dates + weld_times.Start
    weld_times["End_Change"] = dates + weld_times.End
    weld_times.Start_Change = pd.to_datetime(weld_times.Start_Change, format='%d/%m/%y %H:%M:%S.%f')
    weld_times.End_Change = pd.to_datetime(weld_times.End_Change, format='%d/%m/%y %H:%M:%S.%f')
    weld_times = weld_times.drop("id", axis=1)
    weld_times = weld_times.drop("Start", axis=1)
    weld_times = weld_times.drop("End", axis=1)
    weld_times = weld_times.drop("Date", axis=1)
    times = []
    
    maxi = 0
    for row in weld_times.itertuples():
        end_time = row.End_Change
        start_time = row.Start_Change
        diff = (end_time - start_time).seconds
        if  diff < 2 and diff > maxi:
            maxi = (end_time - start_time).seconds
    
    maxi = int(maxi*1400) - 1
    
    for row in weld_times.itertuples():
        #need to make x second clips, so first get the number of milliseconds in the weld
        end_time = row.End_Change
        start_time = row.Start_Change
        if (end_time - start_time).seconds < 2:
            weld = int((end_time - start_time).microseconds/1000)
            remaining = maxi - weld
            random_before = int((np.random.rand(1)*remaining)[0])
            random_end = remaining - random_before
            
            start_sample = start_time - dt.timedelta(microseconds = random_before*1000)
            end_sample = end_time + dt.timedelta(microseconds = random_end*1000)
            times.append([start_sample, end_sample, start_time, end_time])
    return times, maxi


def insert_ones(y, time_start, start_sample, end_sample, Ty, maxi):
    """
    Pass in an array of 0s of length 3 seconds (All of size 1665)
    Corresponds to a filter size of 8 and a stride of 3 for a 3000 item X
    each entry in Y then corresponds to ~ 0.003s 
    Also the start/end times of a weld
    
    This will then turn all of those time steps into 1s instead of 0s and return
    the updated y to append into my examples
    """
    write_from = int(((start_sample-time_start).seconds*1000 + (start_sample-time_start).microseconds/1000)*(Ty/maxi))
    write_to = int(((end_sample-start_sample).seconds*1000 + (end_sample-start_sample).microseconds/1000)*(Ty/maxi)) + write_from
    
    for i in range(Ty):
        if i >= write_from and i <= write_to:
            y[i][0] = 1
            
    return y

def get_xs(time_start, time_end, maxi):
    """
    This will take in the times function and return an array of arrays that 
    has all of the acceleration/ pressure gyro data for the time stamps passed
    in.
    """
    store = pd.HDFStore("Sensor_Data.h5")
    nrows = store.get_storer("Combined").nrows
    prev_chunk=0
    for chunk in range(0, nrows, 1000000):
        #Read in data from HDFStore in chunks to not upset memory
        df = store["Combined"][prev_chunk:chunk]
        #First time this passes it will need a pass
        if len(df) > 0:
            #Look for my start and end times and if they exist do things
            df = df[df.Time.between(time_start,time_end)]
            if len(df) > 0:
                #Using the same code I used in Gather_Files to make sure index
                #and everything else works ok. Need to populate X up to the
                #expected 5000 length, so interpolating
                df_temp = df.copy()
                df_temp.set_index("Time", inplace=True)
                full_times = pd.date_range(time_start, time_end, freq='ms')
                df_del = pd.DataFrame()
                df_del["Time"] = full_times
                df_del.set_index("Time", inplace=True)
                x = df_temp.join(df_del, how='outer')
                x = x.interpolate(limit_direction="both")
                #x["Time"] = x.index -- RNN doesn't take in time values
                x.set_index(pd.Series(range(0,len(x))), inplace=True)
                while len(x) > maxi:
                    x = x.drop(x.index[len(x)-1])
                return x.values
                break
                
        if chunk + 1000000 > nrows:
            df = store["Combined"][chunk:]
            df = df[df.Time.between(time_start,time_end)]
            if len(df) > 0:
                df_temp = df.copy()
                df_temp.set_index("Time", inplace=True)
                full_times = pd.date_range(time_start, time_end, freq='ms')
                df_del = pd.DataFrame()
                df_del["Time"] = full_times
                df_del.set_index("Time", inplace=True)
                x = df_temp.join(df_del, how='outer')
                x = x.interpolate(limit_direction="both")
                #x["Time"] = x.index -- RNN doesn't take in time values
                x.set_index(pd.Series(range(0,len(x))), inplace=True)
                while len(x) > maxi:
                    x = x.drop(x.index[len(x)-1])
                return x.values
                break
        prev_chunk = chunk
    x = pd.DataFrame()
    return x.values