#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:13:30 2020

@author: ogk
"""
import torch
import param
import numpy as np
import new_get_data

def New_Dataloader(session):
    print('session: ', session)
    train_fr, train_label, test_fr, test_label = new_get_data.new_data_preprocessing(session)
    
    train_input = torch.from_numpy(train_fr).type(torch.FloatTensor)
    train_output = torch.from_numpy(train_label).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(train_input, train_output)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = param.batch_size, shuffle=True)
    
    test_input = torch.from_numpy(test_fr).type(torch.FloatTensor)
    test_output = torch.from_numpy(test_label).type(torch.FloatTensor)
    test_dataset = torch.utils.data.TensorDataset(test_input, test_output)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = param.test_bs, shuffle=False)
    
    return train_dataloader, test_dataloader

def Mix_Dataloader(source):
    print('source: ', source)
    print('target: ', source+1)
    S_train_fr, S_train_label, S_test_fr, S_test_label = new_get_data.new_data_preprocessing(source)
    T_train_fr, T_train_label, T_test_fr, T_test_label = new_get_data.new_data_preprocessing(source+1)
    
    train_fr = np.concatenate((S_train_fr, T_train_fr), axis=0)
    train_mov = np.concatenate((S_train_label, T_train_label), axis=0)
    test_fr = T_test_fr
    test_mov = T_test_label
    
    train_input = torch.from_numpy(train_fr).type(torch.FloatTensor)
    train_output = torch.from_numpy(train_mov).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(train_input, train_output)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = param.batch_size, shuffle=True)
    
    test_input = torch.from_numpy(test_fr).type(torch.FloatTensor)
    test_output = torch.from_numpy(test_mov).type(torch.FloatTensor)
    test_dataset = torch.utils.data.TensorDataset(test_input, test_output)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = param.test_bs, shuffle=False)
    
    return train_dataloader, test_dataloader


def Source_dataloader(session):
    print('suorce: ', session)
    train_fr, train_label, test_fr, test_label = new_get_data.new_data_preprocessing(session)
    train_domain = np.zeros(len(train_fr))
    test_domain = np.zeros(len(test_fr))
    
    train_input = torch.from_numpy(train_fr).type(torch.FloatTensor)
    train_output = torch.from_numpy(train_label).type(torch.FloatTensor)
    train_output2 = torch.from_numpy(train_domain).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(train_input, train_output, train_output2)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = param.batch_size, shuffle=True)
    
    test_input = torch.from_numpy(test_fr).type(torch.FloatTensor)
    test_output = torch.from_numpy(test_label).type(torch.FloatTensor)
    train_output2 = torch.from_numpy(test_domain).type(torch.FloatTensor)
    test_dataset = torch.utils.data.TensorDataset(test_input, test_output, train_output2)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = param.test_bs, shuffle=False)
    
    return train_dataloader, test_dataloader
    
def Target_dataloader(session):
    print('target: ', session)
    train_fr, train_label, test_fr, test_label = new_get_data.new_data_preprocessing(session)
    train_domain = np.ones(len(train_fr))
    test_domain = np.ones(len(test_fr))
    
    train_input = torch.from_numpy(train_fr).type(torch.FloatTensor)
    train_output = torch.from_numpy(train_label).type(torch.FloatTensor)
    train_output2 = torch.from_numpy(train_domain).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(train_input, train_output, train_output2)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = param.batch_size, shuffle=True)
    
    test_input = torch.from_numpy(test_fr).type(torch.FloatTensor)
    test_output = torch.from_numpy(test_label).type(torch.FloatTensor)
    train_output2 = torch.from_numpy(test_domain).type(torch.FloatTensor)
    test_dataset = torch.utils.data.TensorDataset(test_input, test_output, train_output2)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = param.test_bs, shuffle=False)
    
    return train_dataloader, test_dataloader






# for s in range(param.num_session):
#     train_fr_order, train_label_order, train_session_label, val_fr_order, val_label_order, val_session_label = get_data.data_preprocess(s)
#     if s == 0:
#         train_fr = train_fr_order
#         train_mov = train_label_order
#         train_class = train_session_label
#         test_fr = val_fr_order
#         test_mov = val_label_order
#         test_class = val_session_label
#     else:
#         train_fr = np.concatenate((train_fr, train_fr_order), axis=0)
#         train_mov = np.concatenate((train_mov, train_label_order), axis=0)
#         train_class = np.concatenate((train_class, train_session_label), axis=0)
#         test_fr = np.concatenate((test_fr, val_fr_order), axis=0)
#         test_mov = np.concatenate((test_mov, val_label_order), axis=0)
#         test_class = np.concatenate((test_class, val_session_label), axis=0)
        
# train_data = torch.from_numpy(train_fr).type(torch.FloatTensor)
# train_motion = torch.from_numpy(train_mov).type(torch.FloatTensor)
# train_label = torch.from_numpy(train_class).type(torch.FloatTensor)
# train_dataset = torch.utils.data.TensorDataset(train_data, train_motion, train_label)
# train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = param.batch_size, shuffle=True)

# test_data = torch.from_numpy(test_fr).type(torch.FloatTensor)
# test_motion = torch.from_numpy(test_mov).type(torch.FloatTensor)
# test_label = torch.from_numpy(test_class).type(torch.FloatTensor)
# test_dataset = torch.utils.data.TensorDataset(test_data, test_motion, test_label)
# test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = 1, shuffle=False)
