"""
Created Fri Sept 7

@Author nsmith

This module is going to be the new way of running the RNN all at once. It will look for
various different things in the folder structure, and if it doesn't find them it will 
execute various other things!
As an example, if there is already an h5 data structure for the sensor data it won't run
that part of the code, etc.
"""

import PreProcess_Data
import Gather_Files
import Build_Examples
import Send_X_to_H5
import RNN

import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

#If the sensor data hasn't been saved into a database let's do that now
cur_path = os.path.dirname(os.path.abspath(__file__))
sensor_name = Path(cur_path + "\Sensor_Data.h5")
if sensor_name.exists():
    print("Sensor data already in h5, next step")
else:
    PreProcess_Data.main(cur_path, "Temp")
    PreProcess_Data.main(cur_path, "Gyro")
    Gather_Files.main()
    
#If there isn't an X h5 made, make that - will also need a times file that is
#containing a list of ONLY times that we need annnnnd yeah that's it.
X_name = Path(cur_path + "\X_Examples.h5")
times_name = Path(cur_path + "\RNN Data Times.txt")
if X_name.exists():
    print("Xs are in an h5, dont need to do anything here")
else:
    if times_name.exists():
        Send_X_to_H5.Send_X_To_h5()
    else:
        print("""You'll need to add a data file to this folder structure with\n
              the name of RNN Data Times.txt\n
              The correct format of that file should be:\n
              UID Time Start Time End Estimated Regime \n
              and all of that should be tab deliminated (4 columns).""")
        exit()
        
#Next we can retreive the new X and Y arrays that will be used for training
#and testing the RNN.
