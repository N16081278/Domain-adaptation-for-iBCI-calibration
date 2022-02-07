#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:24:58 2021

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

for session in range(1, 2):

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
    new_data = Unsort_data
    # make up channel complete
    Channel_List = []
    channel_arr = []
    channel_arr.append([ -1, 42, 46, 25, 31, 35, 39, 41, 47, -2])
    channel_arr.append([ 38, 40, 48, 27, 29, 33, 37, 43,  6, 45])
    channel_arr.append([ 34, 36, 44,  1,  9, 13, 17, 21,  2, 88])
    channel_arr.append([ 30, 32, 89, 93,  5, 15, 19, 23,  8, 84])
    channel_arr.append([ 26, 28, 81, 85, 87, 91,  7,  4, 86, 80])
    channel_arr.append([ 22, 24, 77, 79, 83,  3, 11, 66, 82, 76])
    channel_arr.append([ 18, 20, 73, 75, 95, 54, 62, 74, 78, 72])
    channel_arr.append([ 14, 16, 94, 96, 57, 58, 50, 70, 64, 68])
    channel_arr.append([ 10, 12, 90, 92, 61, 65, 69, 71, 56, 60])
    channel_arr.append([ -3, 51, 49, 53, 55, 59, 63, 67, 52, -4])
    for i in range(10):
        for j in range(10):
            Channel_List.append(channel_arr[i][j])
    channel_arr = np.array(channel_arr)
    all_fr = np.sum(Unsort_data, axis=0)
    lost_num = np.sum(all_fr==0)
    lost_ch = np.where(all_fr==0)[0]
    for i in range(10):
        for j in range(10):
            if channel_arr[i, j] > 0:
                if all_fr[channel_arr[i, j]-1] == 0:
                    print('lost channel: ', channel_arr[i, j])
                    if i-1 < 0 or channel_arr[i-1, j] < 0:
                        c8 = np.zeros((len(Unsort_data), 1))
                    else:
                        c8 = Unsort_data[:, channel_arr[i-1, j]-1].reshape(len(Unsort_data), 1)
                    if i+1 > 9 or channel_arr[i+1, j] < 0:
                        c2 = np.zeros((len(Unsort_data), 1))
                    else:
                        c2 = Unsort_data[:, channel_arr[i+1, j]-1].reshape(len(Unsort_data), 1)
                    if j-1 < 0 or channel_arr[i, j-1] < 0:
                        c4 = np.zeros((len(Unsort_data), 1))
                    else:
                        c4 = Unsort_data[:, channel_arr[i, j-1]-1].reshape(len(Unsort_data), 1)
                    if j+1 > 9 or channel_arr[i, j+1] < 0:
                        c6 = np.zeros((len(Unsort_data), 1))
                    else:
                        c6 = Unsort_data[:, channel_arr[i, j+1]-1].reshape(len(Unsort_data), 1)
                    if i-1 < 0 or j-1 < 0 or channel_arr[i-1, j-1] < 0:
                        c7 = np.zeros((len(Unsort_data), 1))
                    else:
                        c7 = Unsort_data[:, channel_arr[i-1, j-1]-1].reshape(len(Unsort_data), 1)/np.sqrt(2)
                    if i-1 < 0 or j+1 > 9 or channel_arr[i-1, j+1] < 0:
                        c9 = np.zeros((len(Unsort_data), 1))
                    else:
                        c9 = Unsort_data[:, channel_arr[i-1, j+1]-1].reshape(len(Unsort_data), 1)/np.sqrt(2)
                    if i+1 > 9 or j-1 < 0 or channel_arr[i+1, j-1] < 0:
                        c1 = np.zeros((len(Unsort_data), 1))
                    else:
                        c1 = Unsort_data[:, channel_arr[i+1, j-1]-1].reshape(len(Unsort_data), 1)/np.sqrt(2)
                    if i+1 > 9 or j+1 > 9 or channel_arr[i+1, j+1] < 0:
                        c3 = np.zeros((len(Unsort_data), 1))
                    else:
                        c3 = Unsort_data[:, channel_arr[i+1, j+1]-1].reshape(len(Unsort_data), 1)/np.sqrt(2)
                    C = np.concatenate((c8, c2, c4, c6, c7, c9, c1, c3), axis=1)
                    all_c = np.sum(C, axis=0)
                    lost_c = np.sum(all_c!=0)
                    B = (np.sum(C, axis=1))/lost_c
                    new_data[:, channel_arr[i, j]-1] = B
                    
    index_of_lost_ch = []
    for i in range(len(lost_ch)):
        index_of_lost_ch.append(Channel_List.index(lost_ch[i]+1)) # find lost channel
    
    
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