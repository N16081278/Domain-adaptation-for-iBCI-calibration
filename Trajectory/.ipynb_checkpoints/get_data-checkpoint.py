#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:38:51 2020

@author: ogk
"""

import os
import h5py
from os import listdir
import numpy as np
from sklearn import preprocessing
import param
import function

def data_preprocess(session):
    # Define data root directory
    dataset_path = '/home/jovyan/dataset/indy/Sorted_Spike_Dataset/'
    if os.path.isdir(dataset_path) != True:
        print('\n>> Not find dataset folder path: '+dataset_path)
        print('>> Please Check folder_path && Computer working system !\n')
    
    List_File = listdir(dataset_path)
    List_File.sort()
    file_name = List_File[session-1]
    print('\n Source Dataset :',dataset_path+str(file_name))
    mat_file = h5py.File(dataset_path+str(file_name), 'r') # read mat file
    CHANNELS = mat_file[list(mat_file.keys())[1]] # electric channel Info
    CURSOR_POS = mat_file[list(mat_file.keys())[2]] # cursor position
    FINGER_POS = mat_file[list(mat_file.keys())[3]] # finger position
    SPIKES = mat_file[list(mat_file.keys())[4]] # Spike firing time point 這邊是紀錄發生spike的時間點,所以根據bins寬度,要另外整理
    Session_Unit = SPIKES.shape[0] # session unit count ,此一session有幾個被sorting的unit 第一筆皆為hash
    TIMES = mat_file[list(mat_file.keys())[5]] # session t 0.004=4ms
    time_bin = (TIMES[0])[::param.bin_width]
    
    print('== mat file Info == :')
    print('Mat File keys()', list(mat_file.keys()))
    print('Channel count :', CHANNELS)
    print('Cursor position :', CURSOR_POS)
    print('Finger position :', FINGER_POS)
    print('Spike  :', SPIKES)
    print('Time :', TIMES)
    print(' == Session Info. == :')
    print('Session Unit count :', Session_Unit)
    print('observered time bin ('+str(0.004*param.bin_width*1000)+' ms):')
    print('>>',time_bin)
    print('>> time count:',time_bin.shape, ' ; Trial time ', time_bin[-1]- time_bin[0])
    print('')
    
    # finger position shape = time shape
    pos_z = FINGER_POS[0][::param.bin_width]
    pos_x = FINGER_POS[1][::param.bin_width]
    pos_y = FINGER_POS[2][::param.bin_width]
    # get velocity cm => mm
    vel_z = (pos_z[1:] - pos_z[:-1])*(-10)/(0.004*param.bin_width)
    vel_x = (pos_x[1:] - pos_x[:-1])*(-10)/(0.004*param.bin_width)
    vel_y = (pos_y[1:] - pos_y[:-1])*(-10)/(0.004*param.bin_width)
    
       
    # get accelerate
    acc_z = (vel_z[1:] - vel_z[:-1])/ (0.004*param.bin_width)
    acc_x = (vel_x[1:] - vel_x[:-1])/ (0.004*param.bin_width)
    acc_y = (vel_y[1:] - vel_y[:-1])/ (0.004*param.bin_width)
    acc_z = np.concatenate((np.zeros(1), acc_z), axis=0) # be the same shape for vel ,so add zero shape 
    acc_x = np.concatenate((np.zeros(1), acc_x), axis=0) 
    acc_y = np.concatenate((np.zeros(1), acc_y), axis=0) 
    
    
    pos_z = pos_z[1:]
    pos_x = pos_x[1:]
    pos_y = pos_y[1:]
    pos_z = np.resize(pos_z, (len(pos_z), 1))
    pos_x = np.resize(pos_x, (len(pos_x), 1))
    pos_y = np.resize(pos_y, (len(pos_y), 1))
    vel_z = np.resize(vel_z, (len(vel_z), 1))
    vel_x = np.resize(vel_x, (len(vel_x), 1))
    vel_y = np.resize(vel_y, (len(vel_y), 1))
    acc_z = np.resize(acc_z, (len(acc_z), 1))
    acc_x = np.resize(acc_x, (len(acc_x), 1))
    acc_y = np.resize(acc_y, (len(acc_y), 1))
    
    print('>> Position shape :', pos_z.shape)
    print('>> Velocity shape :', vel_z.shape)
    print('>> Accelerate shape :', acc_z.shape)
    
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
    print('Unsort Data shape:', np.shape(Unsort_data))
    
    
    train_fr = []
    val_fr = []
    
    train_fr = Unsort_data[:, :param.training_len]
    val_fr = Unsort_data[:, param.training_len:]
    
        
    print('\n *****train data fr shape: '+str(np.shape(train_fr))+'*****')
    
    
    training_fr = np.transpose(train_fr)
    valing_fr = np.transpose(val_fr)
    min_max_scaler = preprocessing.MinMaxScaler()
    training_fr_nor = min_max_scaler.fit_transform(training_fr)
    training_fr_max = np.max(training_fr, axis = 0)
    # print('training_fr_max = ', training_fr_max)
    training_fr_max[np.where(training_fr_max==0)] = 1
    valing_fr_nor = np.zeros(valing_fr.shape)
    for kk in range(96):
        valing_fr_nor[:, kk] = (valing_fr[:, kk])/training_fr_max[kk]
    position = np.hstack((pos_x, pos_y, pos_z))
    velocity = np.hstack((vel_x, vel_y, vel_z))
    acceleration = np.hstack((acc_x, acc_y, acc_z))
    
    if param.select_label == 'pos':
        label = position
    elif param.select_label == 'vel':
        label = velocity
    elif param.select_label == 'acc': 
        label = acceleration
    elif param.select_label == 'fr':
        label = Unsort_data
    else:
        print(' >> ERROR : for select label')
    
    return label