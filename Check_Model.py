# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:14:38 2018

@author: nicpa

Need to be able to look at my predictions and see if they are good or bad.

To do that need to do some argmax, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

def predictions(test_X,test_Y, model):
    """
    This will return a confusion matrix ready array, as well as the rolled up
    predictions
    """
    preds = model.predict(test_X)
    y_pred = []
    y_test = []
    for i in test_Y:
        a = np.argmax(i,axis=1)
        for j in a:
            y_test.append(j)
            
    for i in preds:
        a = np.argmax(i,axis=1)
        for j in a:
            y_pred.append(j)
    
    return y_pred, y_test, preds


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

def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    