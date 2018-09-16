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
import Data_Setup
import Build_Model

import tensorflow as tf
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
        
#Next I can retrieve my training and test sets. To do this, use the X.h5 as well
#Build examples code, although I might wrap it all in a new module.
times = Build_Examples.insert_examples()
train_X, train_Y, test_X, test_Y = Data_Setup.setup(times)

train_Y = tf.keras.utils.to_categorical(train_Y)
test_Y = tf.keras.utils.to_categorical(test_Y)
"""
This is the actual model part. Now I don't want to build it here, I am going to
make it in the RNN_Model file. From there it will just be returned here
and then executed here. Reason to execute here is there isn't a ton of other 
stuff that needs to happen to be honest.
"""

model = Build_Model.create_model(train_X, 128,0.3)

opt = tf.keras.optimizers.Adam(lr=1e-6, decay=1e-5)
#lr = learning rate
#decay = slowly take smaller steps
model.compile(loss='binary_crossentropy', 
              optimizer=opt,
              metrics = ['accuracy'])

history = model.fit(train_X, train_Y, 
                    batch_size=32, 
                    epochs=80,
                    validation_data=(test_X, test_Y))



























