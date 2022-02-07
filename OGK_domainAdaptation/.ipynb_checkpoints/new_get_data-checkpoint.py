#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:09:33 2021

@author: ogk
"""

import os
import numpy as np
import param
import h5py
import function
from sklearn import preprocessing

def new_data_preprocessing(session):
    global Train_fr, Train_vel, Test_fr, Test_vel
    dataset_path = '/home/jovyan/dataset/indy/Sorted_Spike_Dataset/'
    if os.path.isdir(dataset_path) != True:
        print('\n>> Not find dataset folder path: '+dataset_path)
        print('>> Please Check folder_path && Computer working system !\n')
    
    
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
    fr = Unsort_data
    # make up channel complete
    if param.repair_ch == True:
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
#                         print('lost channel: ', channel_arr[i, j])
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
                        fr[:, channel_arr[i, j]-1] = B
    else:
        fr = Unsort_data
    
    vel_x = (pos_x[1:] - pos_x[:-1])*(-10)/(0.004*param.bin_width)
    vel_y = (pos_y[1:] - pos_y[:-1])*(-10)/(0.004*param.bin_width)
    vel_x = np.resize(vel_x, (len(vel_x), 1))
    vel_y = np.resize(vel_y, (len(vel_y), 1))
    velocity = np.hstack((vel_x, vel_y))
    
    # train & test
    training_fr = fr[:param.training_len, :]
    training_vel = velocity[:param.training_len, :]
    # if param.testing_len > len(fr[param.training_len:, :]):
    #     test_len = len(fr[param.training_len:, :])
    # else:
    #     test_len = param.testing_len
    testing_fr = fr[param.training_len:, :]
    testing_vel = velocity[param.training_len:-1, :] # param.training_len:-1
    
    # normalize fr
    min_max_scaler = preprocessing.MinMaxScaler()
    training_fr_nor = min_max_scaler.fit_transform(training_fr)
    training_fr_nor = np.array(training_fr_nor)
    training_fr_max = np.max(training_fr, axis = 0)
    training_fr_max[np.where(training_fr_max==0)] = 1 # avaiod over 0
    testing_fr_nor = np.zeros(testing_fr.shape)
    for kk in range(96):
        testing_fr_nor[:, kk] = (testing_fr[:, kk])/training_fr_max[kk]
    
    
    # normalize vel
    scaler = preprocessing.StandardScaler()
    training_vel_nor = scaler.fit_transform(training_vel)
    training_vel_mean = np.mean(training_vel, axis = 0)
    training_vel_std = np.std(training_vel, axis = 0)
    training_vel_std[np.where(training_vel_std==0)] = 1 # avaiod over 0
    testing_vel_nor = np.zeros(testing_vel.shape)
    for kk in range(2):
        testing_vel_nor[:, kk] = (testing_vel[:, kk] - training_vel_mean[kk]) / training_vel_std[kk]
    
    
    # order train fr
    training_fr_order = [] ; testing_fr_order = []
    training_vel_order = [] ; testing_vel_order = []
    if param.tap_size == 1:
        training_fr_order = training_fr_nor.reshape(training_fr_nor.shape[0], 1, training_fr_nor.shape[1])
        testing_fr_order = testing_fr_nor.reshape(testing_fr_nor.shape[0], 1, testing_fr_nor.shape[1])
        training_vel_order = training_vel_nor.reshape(training_vel_nor.shape[0], 1, training_vel_nor.shape[1])
        testing_vel_order = testing_vel_nor.reshape(testing_vel_nor.shape[0], 1, testing_vel_nor.shape[1])
    else:
        pre_train_fr_order = [] ; pre_train_vel_order = []
        pre_test_fr_order = [] ; pre_test_vel_order = []
        pre_train_fr_data = [] ; pre_train_vel_data = []
        pre_test_fr_data = [] ; pre_test_vel_data = []
        
        pre_train_fr_order = np.zeros([param.tap_size-1 , training_fr_nor.shape[1]])
        pre_test_fr_order = np.zeros([param.tap_size-1 , testing_fr_nor.shape[1]])
        pre_train_vel_order = np.zeros([param.tap_size-1 , training_vel_nor.shape[1]])
        pre_test_vel_order = np.zeros([param.tap_size-1 , testing_vel_nor.shape[1]])
        
        pre_train_fr_data = np.concatenate((pre_train_fr_order, training_fr_nor), axis = 0)
        pre_test_fr_data = np.concatenate((pre_test_fr_order, testing_fr_nor), axis = 0)
        pre_train_vel_data = np.concatenate((pre_train_vel_order, training_vel_nor), axis = 0)
        pre_test_vel_data = np.concatenate((pre_test_vel_order, testing_vel_nor), axis = 0)
    
        for kk in range(param.tap_size):
            a = pre_train_fr_data[kk:kk+param.training_len] # [kk:(kk-param.tap_size)]
            b = pre_test_fr_data[kk:(kk-param.tap_size)]
            c = pre_train_vel_data[kk:kk+param.training_len]
            d = pre_test_vel_data[kk:(kk-param.tap_size)]
            
            aa = a.reshape(a.shape[0], 1, a.shape[1])
            bb = b.reshape(b.shape[0], 1, b.shape[1])
            cc = c.reshape(c.shape[0], 1, c.shape[1])
            dd = d.reshape(d.shape[0], 1, d.shape[1])
            if kk == 0:
                training_fr_order = aa
                testing_fr_order = bb
                training_vel_order = cc
                testing_vel_order = dd
            else:
                training_fr_order = np.concatenate((training_fr_order, aa), axis=1)
                testing_fr_order = np.concatenate((testing_fr_order, bb), axis=1)
                training_vel_order = np.concatenate((training_vel_order, cc), axis=1)
                testing_vel_order = np.concatenate((testing_vel_order, dd), axis=1)
    
    # split data
    if param.split == True:
        
        avg_v = np.sqrt(np.power(training_vel[:, 0], 2)+np.power(training_vel[:, 1], 2))
        avg_v[avg_v < 5] = 0
        mom_f = np.ones(param.stopping)
        avg_v.resize(len(avg_v))
        avg_v = np.convolve(mom_f, avg_v, 'same')
        avg_v.resize(len(avg_v), 1)
        nonzero = np.nonzero(avg_v)[0]
        stop_t = np.where(avg_v==0)[0]
        
        for i in range(len(nonzero)):
            if i == 0:
                Train_fr = training_fr_order[nonzero[i], :, :].reshape(1, training_fr_order.shape[1], training_fr_order.shape[2])
                Train_vel = training_vel_order[nonzero[i], :, :].reshape(1, training_vel_order.shape[1], training_vel_order.shape[2])
            else:
                Train_fr = np.concatenate([Train_fr, training_fr_order[nonzero[i], :, :].reshape(1, training_fr_order.shape[1], training_fr_order.shape[2])], axis=0)
                Train_vel = np.concatenate([Train_vel, training_vel_order[nonzero[i], :, :].reshape(1, training_vel_order.shape[1], training_vel_order.shape[2])], axis=0)
        
        Test_fr = testing_fr_order
        Test_vel = testing_vel_order
        
    else:
        Train_fr = training_fr_order
        Train_vel = training_vel_order
        Test_fr = testing_fr_order
        Test_vel = testing_vel_order
    
    return Train_fr, Train_vel, Test_fr, Test_vel
