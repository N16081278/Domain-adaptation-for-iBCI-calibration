#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:38:22 2021

@author: ogk
"""

import get_data
import matplotlib.pyplot as plt

path = '/home/ogk/Documents/OGK/Trajectory/trajectory_accumulation/'

for session in range (1, 38):
    position = get_data.data_preprocess(session)
    
    fig, axs = plt.subplots(2, 2, constrained_layout=True, sharex=True, sharey=True)
    # plt.figure(figsize=(20, 16))
    # fig.suptitle('Trajectory with different recording time')
    time = 1 # mins
    duration_1 = int(time*(60*1000)/64)
    axs[0, 0].set_title('1 minute')
    axs[0, 0].plot(position[:duration_1, 0], position[:duration_1, 1])
    axs[0, 0].set_aspect('equal')
    time = 3
    duration_2 = int(time*(60*1000)/64)
    axs[0, 1].set_title('3 minutes')
    axs[0, 1].plot(position[:duration_2, 0], position[:duration_2, 1], 'orange')
    axs[0, 1].set_aspect('equal')
    time = 5
    duration_3 = int(time*(60*1000)/64)
    axs[1, 0].set_title('5 minutes')
    axs[1, 0].plot(position[:duration_3, 0], position[:duration_3, 1], 'green')
    axs[1, 0].set_aspect('equal')
    
    axs[1, 1].set_title('else')
    axs[1, 1].plot(position[duration_3:, 0], position[duration_3:, 1], 'red')
    axs[1, 1].set_aspect('equal')
    
    plt.savefig(path+str(session)+'.png')
    
    plt.cla()
    plt.clf()
    plt.close()

