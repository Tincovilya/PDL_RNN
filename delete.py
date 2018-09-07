# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:07:12 2018

@author: nsmith
"""

import PreProcess_Data
import Gather_Files
import RNN
import Build_Examples
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
"""
cur_path = os.path.dirname(os.path.abspath(__file__))
PreProcess_Data.main(cur_path,"Temp")
PreProcess_Data.main(cur_path,"Gyro")
Gather_Files.main()


plt.figure(figsize=(18,10))
plt.fill_between(x,y1,y2, facecolor='green')
plt.plot(x,y1, linewidth = 0.2, c="k", alpha=0.5)
plt.plot(x,y2, linewidth = 0.2, c="r", alpha=0.5)
plt.show()
"""
times = Build_Examples.insert_examples()

num_filters = [108,78,48,108,78,48,108,78,48,108,78,48,108,78,48]
GRU_units = [23,23,23,13,13,13,20,20,20,30,30,30,33,33,33]
stride = [2,2,2,3,3,3,4,4,4,1,1,1,2,2,2]
kernel_size = [4,6,8,4,6,8,4,6,8,4,6,8,4,6,8]
predictions=[]
vals=[]
masks=[]

for i in range(0, 14):
    Ty = int((int(5000 - kernel_size[i])/stride[i])+1)
    Y = []
    X = []
    for j in tqdm(times):
        Y.append(Build_Examples.insert_ones(np.zeros((Ty,1)), j[0], j[2], j[3],Ty))
        X.append(Build_Examples.get_xs(j[0],j[1]))

    model, train_mask, test_mask, test_X, test_Y = RNN.main(X,Y,Ty,num_filters[i],kernel_size[i],GRU_units[i],stride[i])
    loss, acc = model.evaluate(test_X, test_Y)
    vals.append((loss,acc))
    predictions.append(model.predict_on_batch(test_X))
    model.save("March 20b - " + str(i))
    masks.append((train_mask,test_mask))
    
"""
Ty = 1665
iter=0
for j in predictions2:
    for i in range(Ty):
        if predictions2[iter,i,0] > 0.75:
            predictions2[iter,i,0] = 1
        else:
            predictions2[iter,i,0] = 0
            
    iter+=1

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(range(len(counters)), counters, s=10, c='b', marker="s", label='counters')
ax1.scatter(range(len(y_sum)),y_sum, s=10, c='r', marker="o", label='ysum')
plt.legend(loc='upper left');
plt.show()
"""