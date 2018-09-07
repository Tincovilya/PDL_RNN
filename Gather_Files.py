# -*- coding: utf-8 -*-
"""
Auth: Nicholas Smith
Date: November 15, 2017
Version: 2 (March 7, 2018)

This script will take acceleration (X,Y,Z) and Pressure data, combine them into
one set of CSV files - If a timestamp doesn't exist in Acc/Press I will insert
previous values into that line.
"""
import pandas as pd
import datetime as dt

#Need to gather all acceleration files
def main():
    store = pd.HDFStore('Sensor_Data.h5')
    press_rows = store.get_storer("Temp").nrows
    acc_rows = store.get_storer("Gyro").nrows
    
    if press_rows < 5000000:
        df_press = store["Temp"]
        #both dataframes are small enough to load into memory
        if acc_rows < 5000000:
            df_acc = store["Gyro"]
            df_acc.set_index("Time", inplace=True)
            df_press.set_index("Time", inplace=True)
            df_acc.index = pd.to_datetime(df_acc.index, unit='s')
            df_press.index = pd.to_datetime(df_press.index, unit='s')
            df_pass = df_press.join(df_acc, how="outer")
            df_pass = df_pass.interpolate()
            df_pass["Time"] = df_pass.index
            df_pass.set_index(pd.Series(range(0,len(df_pass))), inplace=True)
            store.append("Combined", df_pass)
        #Need to load acceleration data iteratively
        else:
            i=0
            while i < acc_rows:
                if i+5000000 > acc_rows:
                    load_to = acc_rows-1
                else:
                    load_to = i+5000000
                df_acc = store["Gyro"][i:load_to]
                max_time_acc = df_acc["Time"][len(df_acc)-1]
                min_time_acc = df_acc["Time"].head()[0]
                df_temp_press = df_press[df_press.Time.between(max_time_acc,min_time_acc)].copy()
                df_acc.set_index("Time", inplace=True)
                df_temp_press.set_index("Time", inplace=True)
                df_acc.index = pd.to_datetime(df_acc.index, unit='s')
                df_temp_press.index = pd.to_datetime(df_temp_press.index, unit='s')
                df_pass = df_temp_press.join(df_acc, how="outer")
                df_pass = df_pass.interpolate()
                df_pass["Time"] = df_pass.index
                df_pass.set_index(pd.Series(range(0,len(df_pass))), inplace=True)
                store.append("Combined", df_pass)
                del df_acc
                del df_temp_press
                del df_pass
                i+=5000000
    else:
        j=0
        while j < press_rows:
            if j+5000000 > press_rows:
                load_to_press = press_rows-1
            else:
                load_to_press = j+5000000
            df_press = store["Temp"][j:load_to_press]
            max_time_press = df_press["Time"][len(df_press)-1]
            max_time_acc = dt.datetime(2099,1,1)
            while max_time_press.to_pydatetime() < max_time_acc:
                if i+5000000 > acc_rows:
                    load_to = acc_rows-1
                else:
                    load_to = i+5000000
                df_acc = store["Gyro"][i:load_to]
                max_time_acc = df_acc["Time"][len(df_acc)-1]
                min_time_acc = df_acc["Time"].head()[0]
                df_temp_press = df_press[df_press.Time.between(max_time_acc,min_time_acc)].copy()
                df_acc.set_index("Time", inplace=True)
                df_temp_press.set_index("Time", inplace=True)
                df_acc.index = pd.to_datetime(df_acc.index, unit='s')
                df_temp_press.index = pd.to_datetime(df_temp_press.index, unit='s')
                df_pass = df_temp_press.join(df_acc, how="outer")
                df_pass = df_pass.interpolate()
                df_pass["Time"] = df_pass.index
                df_pass.set_index(pd.Series(range(0,len(df_pass))), inplace=True)
                store.append("Combined", df_pass)
                del df_acc
                del df_temp_press
                del df_pass
                i+=5000000
            j+=5000000
            
    store.close()