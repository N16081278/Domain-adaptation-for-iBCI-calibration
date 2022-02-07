#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:21:34 2021

@author: ogk
"""


import os
import h5py
import param
import numpy as np
import matplotlib.pyplot as plt

save_path = '/home/ogk/Documents/OGK/Trajectory/new_label_v/5000/'

dataset_path = '/home/ogk/Documents/dataset/data/data__I_download/'
if os.path.isdir(dataset_path) != True:
    print('\n>> Not find dataset folder path: '+dataset_path)
    print('>> Please Check folder_path && Computer working system !\n')
    
stopping = 50
# session = 8
for session in range(1, 38):
    print('session=', session)
    List_File = os.listdir(dataset_path)
    List_File.sort()
    file_name = List_File[session-1]
    print('\n Dataset :',dataset_path+str(file_name))
    mat_file = h5py.File(dataset_path+str(file_name), 'r') # read mat 
    FINGER_POS = mat_file[list(mat_file.keys())[3]]
    pos_x = FINGER_POS[1][::param.bin_width]
    pos_y = FINGER_POS[2][::param.bin_width]
    
    TIMES = mat_file[list(mat_file.keys())[5]]
    time_bin = (TIMES[0])[::param.bin_width]
    time_bin = time_bin[:-1]
    
    vel_x = (pos_x[1:] - pos_x[:-1])*(-10)/(0.004*param.bin_width)
    vel_y = (pos_y[1:] - pos_y[:-1])*(-10)/(0.004*param.bin_width)
    vel_x = np.resize(vel_x, (len(vel_x), 1))
    vel_y = np.resize(vel_y, (len(vel_y), 1))
    velocity = np.hstack((vel_x, vel_y))
    
    avg_v = np.sqrt(np.power(vel_x, 2)+np.power(vel_y, 2))
    avg_v[avg_v < 5] = 0
    mom_f = np.ones(stopping)
    avg_v.resize(len(avg_v))
    avg_v = np.convolve(mom_f, avg_v, 'same')
    avg_v.resize(len(avg_v), 1)
    nonzero = np.nonzero(avg_v)
    stop_t1 = np.where(avg_v==0)[0]
    stop_t2 = np.where(avg_v!=0)[0]
    fuck1 = stop_t1[1:] - stop_t1[:-1] - 1
    fuck2 = stop_t2[1:] - stop_t2[:-1] - 1
    timing1 = np.where(fuck1!=0)[0] + 1
    timing2 = np.where(fuck2!=0)[0] + 1
    ttt1 = []
    for i in timing1:
        ttt1.append(stop_t1[i])
    ttt2 = []
    for i in timing2:
        ttt2.append(stop_t2[i])
    
    # if len(ttt2) > 3:
    #     if len(ttt1) == len(ttt2):
    #         ttt = np.array([ttt1, ttt2]).T
    #         ttt = np.vstack([[0, timing1[0]], ttt])
    #     else:
    #         ttt = np.array([ttt1, ttt2[1:]]).T
        
    # stoping_t = np.zeros(ttt.shape)
    # for i in range(2):
    #     for j in range(len(ttt)):
    #         stoping_t[j, i] = time_bin[ttt[j, i]]
    
    new_v = []
    for i in nonzero[0]:
        new_v.append(velocity[i, :])
    new_v = np.array(new_v)
    end_t = 0.064*(len(new_v))
    new_t = np.arange(0, end_t, 0.064)
    
    
    time = 1 # mins
    duration_1 = int(time*(60*1000)/64)
    time = 3 # mins
    duration_3 = int(time*(60*1000)/64)
    time = 5 # mins
    duration_5 = int(time*(60*1000)/64)
    
    
    t = time_bin
    v = velocity
    plt.figure(figsize=(16, 10))
    plt.margins(x=0)
    plt.grid(True)
    plt.title('Session '+str(session), fontsize=24)
    plt.plot(t, v[:, 0], linewidth=0.5, alpha=0.7)
    plt.plot(t, v[:, 1], linewidth=0.5, alpha=0.7)
    plt.legend(['Vx', 'Vy'], fontsize=24, loc='upper right')
    plt.xlabel('Time(sec)', fontsize=20)
    plt.ylabel('Velocity(mm/sec)', fontsize=20)
    plt.axvline(t[duration_1], color='g')
    plt.axvline(t[duration_3], color='g')
    plt.axvline(t[duration_5], color='g')
    plt.text(t[duration_1]-20, -400, r'1 min', fontdict={'size':'20','color':'b'})
    plt.text(t[duration_3]-20, -400, r'3 mins', fontdict={'size':'20','color':'b'})
    plt.text(t[duration_5]-20, -400, r'5 mins', fontdict={'size':'20','color':'b'})
    
    
    if len(ttt1) == len(ttt2):
        if len(ttt1) > 0 or len(ttt2) > 0:
            ttt = np.array([ttt1, ttt2]).T
            ttt = np.vstack([[0, timing1[0]], ttt])
        else:
            if len(stop_t1) > 0 and len(stop_t2) > 0:
                ttt = np.array([[stop_t1[0]], [stop_t2[0]]]).T
            else:
                ttt = np.array([[0, 0]])
    elif len(ttt1) < len(ttt2):
        ttt1 = [timing2[0], *ttt1]
        
        ttt = np.array([ttt1, ttt2]).T
        # ttt = np.vstack([[0, timing1[0]], ttt])
    else:
        # ttt2 = [timing1[0], *ttt2]
        ttt = np.array([ttt1[:-1], ttt2]).T
        ttt = np.vstack([[0, timing1[0]], ttt])
    
    
    stoping_t = np.zeros(ttt.shape)
    for i in range(2):
        for j in range(len(ttt)):
            stoping_t[j, i] = time_bin[ttt[j, i]]

    for i in range(len(stoping_t)):
        plt.axvspan(stoping_t[i, 0], stoping_t[i, 1], color='red', alpha=0.3)
    plt.xlim(t[0], t[5000])
    # plt.ylim(0, 1000)
    
    plt.savefig(save_path+str(session)+'.png')
    plt.cla()
    plt.clf()
    plt.close()