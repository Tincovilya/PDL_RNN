# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:24:06 2018

@author: nsmith
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
"""
training = []
testing = []
for i in masks:
    training.append(i[0])
    testing.append(i[1])

dict_train={}
dict_test={}
for i in range(0, 391):
    dict_train.update({i:0})
    dict_test.update({i:0})

for i in training[4]:
        #for j in i:
        dict_train[i] = dict_train[i] + 1
        
for i in training[4]:
        #for j in i:
        dict_test[i] = dict_test[i] + 1
        
plt.scatter(range(len(dict_train)), dict_train.values(), s=0.5)

plt.scatter(range(len(dict_test)), dict_test.values(), s=0.5)

#for i in range(0, 391):
#    if dict_train[i] == 0:
#        print("Never used for training " + str(i))
#    if dict_test[i] == 0:
 #       print("Never used for testing " + str(i))

plt.figure(figsize=(10,10))
plt.imshow(similar,interpolation="none",cmap='Blues')
for (i, j), z in np.ndenumerate(similar):
    plt.text(j, i, round(z,2), ha='center', va='center')
plt.show()
"""
fig = plt.figure(figsize=(15,30))
gs1 = gridspec.GridSpec(7,2)
gs1.update(left=0,right=1,top=0.8, bottom=0)
ax1 = fig.add_subplot(gs1[0,0])
ax1.scatter(range(len(predictions[0][0])), predictions[0][4], s=0.5)
ax2 = fig.add_subplot(gs1[1,0])
ax2.scatter(range(len(predictions[1][0])), predictions[1][4], s=0.5)
ax3 = fig.add_subplot(gs1[2,0])
ax3.scatter(range(len(predictions[2][0])), predictions[2][4], s=0.5)
ax4 = fig.add_subplot(gs1[3,0])
ax4.scatter(range(len(predictions[3][0])), predictions[3][4], s=0.5)
ax5 = fig.add_subplot(gs1[4,0])
ax5.scatter(range(len(predictions[4][0])), predictions[4][4], s=0.5)
ax6 = fig.add_subplot(gs1[5,0])
ax6.scatter(range(len(predictions[5][0])), predictions[5][4], s=0.5)
ax7 = fig.add_subplot(gs1[6,0])
ax7.scatter(range(len(predictions[6][0])), predictions[6][4], s=0.5)
ax8 = fig.add_subplot(gs1[0,1])
ax8.scatter(range(len(predictions[7][0])), predictions[7][4], s=0.5)
ax9 = fig.add_subplot(gs1[1,1])
ax9.scatter(range(len(predictions[8][0])), predictions[8][4], s=0.5)
ax10 = fig.add_subplot(gs1[2,1])
ax10.scatter(range(len(predictions[9][0])), predictions[9][4], s=0.5)
ax11 = fig.add_subplot(gs1[3,1])
ax11.scatter(range(len(predictions[10][0])), predictions[10][4], s=0.5)
ax12 = fig.add_subplot(gs1[4,1])
ax12.scatter(range(len(predictions[11][0])), predictions[11][4], s=0.5)
ax13 = fig.add_subplot(gs1[5,1])
ax13.scatter(range(len(predictions[12][0])), predictions[12][4], s=0.5)
ax14 = fig.add_subplot(gs1[6,1])
ax14.scatter(range(len(predictions[13][0])), predictions[13][4], s=0.5)
