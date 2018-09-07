# -*- coding: utf-8 -*-
"""
Auth: Nicholas Smith
Date: March 5, 2018
Version: 2

This script will take acceleration (X,Y,Z) or Pressure/Temp data and combine
them into one file. I really only want very sepcific things from these files,
ie only data that is valid for the actual launch - receive. Then I am going to
test the total size of my dataframe, if it exceeds a certain size I know that
other things will have to happen because it simply will not be able to be held
in memory.

Also note that launch/receive times will be pulled from a CSV called "AGMs"
any times out side of this window will not be used.
"""
import pandas as pd
import glob
import datetime as dt
import numpy as np
from tqdm import tqdm

#Need to gather all acceleration files
def main(path, file_type):
    allFiles = glob.glob(path + "\\" + file_type + "*.csv")
    
    dfPassages = pd.read_csv(path+"\\AGMs.csv",index_col=None, header=0)
    dfPassages.columns = ["Name","Chainage","Elevation","Day","Time","Type"]
    Launch_Time = str(dfPassages["Day"][0]) + " " + str(dfPassages["Time"][0])
    Launch_Time = dt.datetime.strptime(Launch_Time, '%B %d, %Y %I:%M:%S %p')
    L_Time = dt.datetime.strftime(Launch_Time, '%Y-%m-%d %H:%M:%S.%f')
    Launch_Time = dt.datetime.strptime(L_Time, '%Y-%m-%d %H:%M:%S.%f')
    Receive_Time = str(dfPassages["Day"][len(dfPassages)-1]) + " " + str(dfPassages["Time"][len(dfPassages)-1])
    Receive_Time = dt.datetime.strptime(Receive_Time, '%B %d, %Y %I:%M:%S %p')
    R_Time = dt.datetime.strftime(Receive_Time, '%Y-%m-%d %H:%M:%S.%f')
    Receive_Time = dt.datetime.strptime(R_Time, '%Y-%m-%d %H:%M:%S.%f')
    
    list_ = []
    row_size = 0
    for filename in tqdm(allFiles):
        if file_type == "Temp":
            df = pd.read_csv(filename,index_col=None, header=0,names=["Time","Temp","Pressure","No-Name"], 
                             dtype={"Temp":np.float64,"Pressure":np.float64})
            df.Time = pd.to_datetime(df.Time, format='%Y-%m-%d %H:%M:%S.%f')
            row_size+=len(df)
            if df.Time[0] > Launch_Time and df.Time[len(df)-1] < Receive_Time:
                df.drop(["No-Name"], axis=1,inplace=True)
                df.drop(["Temp"], axis=1,inplace=True)
                list_.append(df)
            elif df.Time[0] < Launch_Time and df.Time[len(df)-1] < Receive_Time:
                max_time = df.Time[len(df)-1]
                df = df[(df['Time'] >= Launch_Time) & (df['Time'] <= max_time)]
                df.drop(["No-Name"], axis=1,inplace=True)
                df.drop(["Temp"], axis=1,inplace=True)
                list_.append(df)
            elif df.Time[0] > Launch_Time and df.Time[len(df)-1] > Receive_Time:
                min_time = df.Time[0]
                df = df[(df['Time'] >= min_time) & (df['Time'] <= Receive_Time)]
                df.drop(["No-Name"], axis=1,inplace=True)
                df.drop(["Temp"], axis=1,inplace=True)
                list_.append(df)
        else:
            df = pd.read_csv(filename,index_col=None, header=0, names=["Time","Gyro1","Gyro2","Gyro3","Acc_X","Acc_Y","Acc_Z","No-Name"],
                             dtype={"Gyro1":np.float64,"Gyro2":np.float64,"Gyro3":np.float64,"Acc_X":np.float64,"Acc_Y":np.float64,"Acc_Z":np.float64})
            df.columns = ["Time","Gyro1","Gyro2","Gyro3","Acc_X","Acc_Y","Acc_Z","No-Name"]
            df.Time = pd.to_datetime(df.Time,format='%Y-%m-%d %H:%M:%S.%f')
            if df.Time[0] > Launch_Time and df.Time[len(df)-1] < Receive_Time:
                df.drop(["No-Name"], axis=1,inplace=True)
                list_.append(df)
            elif df.Time[0] < Launch_Time and df.Time[len(df)-1] < Receive_Time:
                max_time = df.Time[len(df)-1]
                df = df[(df['Time'] >= Launch_Time) & (df['Time'] <= max_time)]
                df.drop(["No-Name"], axis=1,inplace=True)
                list_.append(df)
            elif df.Time[0] > Launch_Time and df.Time[len(df)-1] > Receive_Time:
                min_time = df.Time[0]
                df = df[(df['Time'] >= min_time) & (df['Time'] <= Receive_Time)]
                df.drop(["No-Name"], axis=1,inplace=True)
                list_.append(df)
                
        if row_size > 5000000:
            store = pd.HDFStore('Sensor_Data.h5')
            df_pass = pd.concat(list_, ignore_index=True)
            try:
                nrows = store.get_storer(file_type).nrows
            except:
                nrows = 0
            df_pass = df_pass.set_index(pd.Series(df_pass.index) + nrows)
            store.append(file_type, df_pass)
            list_ = []
            del df_pass
            row_size = 0
            store.close()
    
    store = pd.HDFStore('Sensor_Data.h5')
    try:
        df_pass = pd.concat(list_, ignore_index=True)
    except:
        df_pass = df
        
    try:
        nrows = store.get_storer(file_type).nrows
    except:
        nrows = 0
    df_pass = df_pass.set_index(pd.Series(df_pass.index) + nrows)
    store.append(file_type, df_pass)
    store.close()