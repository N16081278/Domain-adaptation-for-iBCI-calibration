#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:00:11 2021

@author: ogk
"""

import os
import h5py
import param
import function
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/home/ogk/Documents/dataset/data/data__I_download/'
if os.path.isdir(dataset_path) != True:
    print('\n>> Not find dataset folder path: '+dataset_path)
    print('>> Please Check folder_path && Computer working system !\n')

save_path = '/home/ogk/Documents/OGK/Trajectory/new_label_v/new/'

stopping = 50

for session in range(1, 38):

    print('\nSession=', session)
    List_File = os.listdir(dataset_path)
    List_File.sort()
    file_name = List_File[session-1]
    print('Dataset :',dataset_path+str(file_name))
    mat_file = h5py.File(dataset_path+str(file_name), 'r') # read mat 
    FINGER_POS = mat_file[list(mat_file.keys())[3]]
    SPIKES = mat_file[list(mat_file.keys())[4]]
    Session_Unit = SPIKES.shape[0]
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
    stop_t = np.where(avg_v==0)[0]
    
    # get firing rate
    search_index = 0
    for ch_index in param.channel:
        # concat data by channel
        fr_unit_hash = []
        fr_unit_01 = []
        fr_unit_02 = []
        fr_unit_03 = []
        fr_unit_04 = []
        
        unit_hash = mat_file[SPIKES[0][ch_index-1]] 
        unit_01 = mat_file[SPIKES[1][ch_index-1]]
        unit_02 = mat_file[SPIKES[2][ch_index-1]]
    
        # unit.shape[0] == 2 ,.mat file show this is null array>>表示 .mat檔對應的是空集合的channel
        if unit_hash.shape[0] != 2 :
            fr_unit_hash = function.Get_Spike_Firring(unit_hash[0], time_bin)
        else:
            fr_unit_hash = np.zeros([1,time_bin.shape[0]-1])
    
        if unit_01.shape[0] != 2 :
            fr_unit_01 = function.Get_Spike_Firring(unit_01[0], time_bin)
        else:
            fr_unit_01 = np.zeros([1,time_bin.shape[0]-1]) 
    
        if unit_02.shape[0] !=2 :    
            fr_unit_02 = function.Get_Spike_Firring(unit_02[0], time_bin)
        else:
            fr_unit_02 = np.zeros([1,time_bin.shape[0]-1])
    
        if Session_Unit == 5:
            unit_03 = mat_file[SPIKES[3][ch_index-1]]
            unit_04 = mat_file[SPIKES[4][ch_index-1]]
            if unit_03.shape[0] !=2 : 
                fr_unit_03 = function.Get_Spike_Firring(unit_03[0], time_bin)
            else:
                fr_unit_03 = np.zeros([1,time_bin.shape[0]-1])
    
            if unit_04.shape[0] !=2 :
                fr_unit_04 = function.Get_Spike_Firring(unit_04[0], time_bin)
            else:
                fr_unit_04 = np.zeros([1,time_bin.shape[0]-1])
        else:
            fr_unit_03 = np.zeros([1,time_bin.shape[0]-1])
            fr_unit_04 = np.zeros([1,time_bin.shape[0]-1])
            
        # concate unit data
        if search_index == 0:
            Firing_rate_hash = fr_unit_hash
            Firing_rate_unit_1 = fr_unit_01
            Firing_rate_unit_2 = fr_unit_02
            Firing_rate_unit_3 = fr_unit_03
            Firing_rate_unit_4 = fr_unit_04
        else:
            Firing_rate_hash = np.concatenate((Firing_rate_hash, fr_unit_hash), axis=0)
            Firing_rate_unit_1 = np.concatenate((Firing_rate_unit_1, fr_unit_01), axis=0)
            Firing_rate_unit_2 = np.concatenate((Firing_rate_unit_2, fr_unit_02), axis=0)
            Firing_rate_unit_3 = np.concatenate((Firing_rate_unit_3, fr_unit_03), axis=0)
            Firing_rate_unit_4 = np.concatenate((Firing_rate_unit_4, fr_unit_04), axis=0)
        search_index +=1
    
    # print('=====',np.all(Firing_rate_unit_4 == 0))
    Unsort_data = Firing_rate_hash+ Firing_rate_unit_1+ Firing_rate_unit_2+ Firing_rate_unit_3 +Firing_rate_unit_4
    Unsort_data = np.transpose(Unsort_data)
    print('Unsort Data shape:', np.shape(Unsort_data))
    
    new_v = []
    new_fr = []
    for i in nonzero[0]:
        new_v.append(velocity[i, :])
        new_fr.append(Unsort_data[i-1, :])
    new_v = np.array(new_v)
    new_fr = np.array(new_fr)
    end_t = 0.064*(len(new_v))
    new_t = np.arange(0, end_t-0.01, 0.064)
    
    # make up channel complete
    all_fr = np.sum(new_fr, axis=0)
    lost_num = np.sum(all_fr==0)
    lost_ch = np.where(all_fr==0)[0]
    ok_ch = np.nonzero(all_fr)[0]
    nonlost_ch = np.zeros([len(new_fr), len(ok_ch)])
    for i in range(len(ok_ch)):
        nonlost_ch[:, i] = new_fr[:, ok_ch[i]]
    avg_fr = np.mean(nonlost_ch, axis=1)
    for i in lost_ch:
        new_fr[:, i] = avg_fr
    
    # np.savez(save_path+'S'+str(session), fr=new_fr, t=new_t, vel=new_v)
        
    time = 1 # mins
    duration_1 = int(time*(60*1000)/64)
    time = 3 # mins
    duration_3 = int(time*(60*1000)/64)
    time = 5 # mins
    duration_5 = int(time*(60*1000)/64)
    
    t = new_t
    v = new_v
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

    plt.savefig(save_path+str(session)+'.png')

    plt.cla()
    plt.clf()
    plt.close()