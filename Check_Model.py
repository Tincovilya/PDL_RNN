# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:14:38 2018

@author: nicpa

Need to be able to look at my predictions and see if they are good or bad.

To do that need to do some argmax, etc.
"""

import numpy as np
import matplotlib.pyplot as plt

def predictions(test_X,model):
    predictions = model.predict(test_X)
    arg_max_Y = []
    for i in predictions:
        arg_max_Y.append(np.argmax(i,axis=1))
        
    return arg_max_Y


def plot_history(histories, key='acc'):
  plt.figure(figsize=(16,10))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_acc'],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  
"""
 Call like this
  plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])  
"""
