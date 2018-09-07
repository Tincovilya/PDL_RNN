# -*- coding: utf-8 -*-
"""
Auth: Nicholas Smith
Date: March 5, 2018
Version: 2

This module will plot elevation, pressure, and velocity against 
the overall chainage of a pipeline and also the AGM numbers
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def main(pressure, velocity, AGM_List,title):
    max_dist = max(AGM_List["Chainage"])
    
    fig = plt.figure(figsize=(15,10))
    fig.suptitle(title, fontsize = 20)
    gs1 = gridspec.GridSpec(3,1)
    gs1.update(left=0,right=1,top=0.8, bottom=0)
    ax1 = fig.add_subplot(gs1[0,0])
    ax1.axes.get_xaxis().set_visible(False)
    ax1.plot(AGM_List["Chainage"], AGM_List["Elevation"])
    ax1.set_ylabel("MSL (m)", fontsize=14)
    
    ax2 = ax1.twiny()
    new_tick_labels = []
    k=0
    last_label = 0
    for i in AGM_List["Type"]:
        diff = (AGM_List["Chainage"][k] - AGM_List["Chainage"][last_label])/max_dist
        if i.lower() == "tap" or i.lower() == "valve":
            if diff > 0.01 or k == 0 or k==len(AGM_List["Type"])-1:
                if k==len(AGM_List["Type"])-1 and diff < 0.01:
                    new_tick_labels[k-1] = ""
                    new_tick_labels.append(AGM_List["AGM"][k])
                elif k==len(AGM_List["Type"])-1 and diff > 0.01:
                    new_tick_labels.append(AGM_List["AGM"][k])
                new_tick_labels.append(AGM_List["AGM"][k])
                last_label=k
            else:
                new_tick_labels.append("")
        elif diff > 0.03 and AGM_List["Type"][k+1].lower != "tap" and AGM_List["Type"][k+1].lower != "valve":
            new_tick_labels.append(AGM_List["AGM"][k])
            last_label=k
        else:
            new_tick_labels.append("")
        k+=1
    new_tick_labels = np.asarray(new_tick_labels)
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(AGM_List["Chainage"])
    ax2.set_xticklabels(new_tick_labels, rotation=90)
    ax2.set_xlabel("AGMs",fontsize=14)
    
    P_mean = pressure["Pressure"].mean()
    P_max = pressure["Pressure"].max()
    ax3 = fig.add_subplot(gs1[1,0])
    ax3.set_ylabel("Pressure (kpa)",fontsize=14)
    ax3.set_ylim([P_mean*0.95,P_max*1.05])
    ax3.axes.get_xaxis().set_visible(False)
    ax3.plot(np.linspace(0,max_dist,len(pressure["Pressure"])), pressure["Pressure"])
    
    #V_min = min(velocity)
    #V_max = max(velocity)
    V_min = 0
    V_max = 5
    ax4 = fig.add_subplot(gs1[2,0])
    ax4.set_ylabel("Velocity (m/s)",fontsize=14)
    ax4.set_ylim([V_min*0.95,V_max*1.05])
    ax4.set_xlabel("Chainage (m)",fontsize=14)
    ax4.plot(np.linspace(0,max_dist,len(velocity)),velocity)
    plt.show()
    
    fig.savefig("triple.pdf", bbox_inches='tight')
