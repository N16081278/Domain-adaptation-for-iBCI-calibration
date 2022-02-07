#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 03:57:35 2021

@author: ogk
"""

import os
import h5py
import param
import numpy as np
import matplotlib.pyplot as plt

save_path = '/home/ogk/Documents/OGK/Trajectory/label_v/'

dataset_path = '/home/ogk/Documents/dataset/data/data__I_download/'
if os.path.isdir(dataset_path) != True:
    print('\n>> Not find dataset folder path: '+dataset_path)
    print('>> Please Check folder_path && Computer working system !\n')

for session in range(8, 9):    
    List_File = os.listdir(dataset_path)
    List_File.sort()
    file_name = List_File[session-1]
    print('\n Dataset :',dataset_path+str(file_name))
    mat_file = h5py.File(dataset_path+str(file_name), 'r') # read mat 
    FINGER_POS = mat_file[list(mat_file.keys())[3]]
    pos_x = FINGER_POS[0][::param.bin_width]
    pos_y = FINGER_POS[1][::param.bin_width]
    
    TIMES = mat_file[list(mat_file.keys())[5]]
    time_bin = (TIMES[0])[::param.bin_width]
    time_bin = time_bin[:-1]
    
    vel_x = (pos_x[1:] - pos_x[:-1])*(-10)/(0.004*param.bin_width)
    vel_y = (pos_y[1:] - pos_y[:-1])*(-10)/(0.004*param.bin_width)
    vel_x = np.resize(vel_x, (len(vel_x), 1))
    vel_y = np.resize(vel_y, (len(vel_y), 1))
    velocity = np.hstack((vel_x, vel_y))
    
    time = 1 # mins
    duration_1 = int(time*(60*1000)/64)
    time = 3 # mins
    duration_3 = int(time*(60*1000)/64)
    time = 5 # mins
    duration_5 = int(time*(60*1000)/64)
    
    plt.figure(figsize=(16, 10))
    plt.margins(x=0)
    plt.grid(True)
    plt.title('Session '+str(session), fontsize=24)
    plt.plot(time_bin[:5000], vel_x[:5000], linewidth=0.5, alpha=0.7)
    plt.plot(time_bin[:5000], vel_y[:5000], linewidth=0.5, alpha=0.7)
    plt.legend(['Vx', 'Vy'], fontsize=24, loc='upper right')
    plt.ylim((-20, 20))
    plt.xlabel('Time(sec)', fontsize=20)
    plt.ylabel('Velocity(mm/sec)', fontsize=20)
    plt.axvline(time_bin[duration_1], color='g')
    plt.axvline(time_bin[duration_3], color='g')
    plt.axvline(time_bin[duration_5], color='g')
    # plt.text(time_bin[duration_1]-30, -400, r'1 min', fontdict={'size':'20','color':'b'})
    # plt.text(time_bin[duration_3]-30, -400, r'3 min', fontdict={'size':'20','color':'b'})
    # plt.text(time_bin[duration_5]-30, -400, r'5 min', fontdict={'size':'20','color':'b'})
    
    # plt.savefig(save_path+str(session)+'.png')
    
    # plt.cla()
    # plt.clf()
    # plt.close()