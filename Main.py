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
import Check_Model

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import os
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

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

#Number of welds vs. no welds:
temp_Y = np.reshape(train_Y, (train_Y.shape[0]*train_Y.shape[1],))

weight = len(temp_Y)/np.sum(temp_Y)
sample_weights = train_Y*27 + 1
sample_weights = sample_weights.reshape(312,3000)

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
              metrics = [precision_threshold(0.8),recall_threshold(0.8), 'accuracy'],
              sample_weight_mode="temporal")

NAME = f"Testing_Model - {int(time.time())}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
#calculate the weighting to use for an imbalanced data set
class_weights = {0:1.,1:weight}

history = model.fit(train_X, train_Y,
                    sample_weight=sample_weights,
                    batch_size=32, 
                    epochs=50,
                    validation_data=(test_X, test_Y),
                    callbacks=[tensorboard])

y_pred, y_test, preds = Check_Model.predictions(test_X,test_Y, model)

Check_Model.plot_history([('Baseline', history)])

"""
Ok so the idea here will be to use a confusion matrix to get results.
First I need to change the [0,1] vectors into [1] vectors. This is just because
it is confused (haha) on how to plot this. Needs my data to be Yes, No, or
whatever.

This code needs to be run after the above because the confusion matrix will plot
really weird if you leave it like this - just copy and paste and run.

cnf_matrix = confusion_matrix(y_test, y_pred)

Check_Model.confusion_matrix(cnf_matrix, classes=["No Weld", "Weld"], normalize=True,
                      title='Normalized confusion matrix')
"""




















