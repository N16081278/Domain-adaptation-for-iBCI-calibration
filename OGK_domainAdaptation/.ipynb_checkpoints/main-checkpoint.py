#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:53:35 2020

@author: ogk
"""

import torch
import train
import dataloader
import model
import numpy as np
from utils import get_free_gpu
import testt
import seaborn as sns
import matplotlib.pyplot as plt
import param

# fixed 1
if param.mode == 'fix':
    train_loader, val_loader = dataloader.New_Dataloader(1)
    
    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        decoder = model.Decoder().cuda()
        time = train.GRU_train(encoder, decoder, train_loader)
    else:
        print("There is no GPU -_-!")
        
    for session in range(1, 38):
        train_loader, val_loader = dataloader.New_Dataloader(session)
        
        encoder = model.Extractor().cuda()
        encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
        decoder = model.Decoder().cuda()
        decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
        
        r2_x, r2_y = testt.tester(encoder, decoder, val_loader)
        print("R2x : ", r2_x)
        print("R2y : ", r2_y)
        a = [[r2_x, r2_y]]
        if session == 1:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)

elif param.mode == 'retrain':
    for session in range(1, 38):
        train_loader, test_loader = dataloader.New_Dataloader(session)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            decoder = model.Decoder().cuda()
            time = train.GRU_train(encoder, decoder, train_loader)
        else:
            print("There is no GPU -_-!")
            
        r2_x, r2_y = testt.tester(encoder, decoder, test_loader)
        print("R2x : ", r2_x)
        print("R2y : ", r2_y)
        a = [[r2_x, r2_y, time]]
        if session == 1:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
            
elif param.mode == 'mix':
    for session in range(1, 37):
        train_loader, test_loader = dataloader.Mix_Dataloader(session)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            decoder = model.Decoder().cuda()
            time = train.GRU_train(encoder, decoder, train_loader)
        else:
            print("There is no GPU -_-!")
            
        r2_x, r2_y = testt.tester(encoder, decoder, test_loader)
        print("R2x : ", r2_x)
        print("R2y : ", r2_y)
        a = [[r2_x, r2_y, time]]
        if session == 1:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
        
elif param.mode == 'tune':
    train_loader, val_loader = dataloader.New_Dataloader(1)
    
    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        decoder = model.Decoder().cuda()
        time = train.GRU_train(encoder, decoder, train_loader)
    else:
        print("There is no GPU -_-!")
        
    for target in range(2, 38):
        T_tune_loader, T_test_loader = dataloader.New_Dataloader(target)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
            decoder = model.Decoder().cuda()
            decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
            total_time = train.GRU_train(encoder, decoder, T_tune_loader)
        else:
            print("There is no GPU -_-!")
        
        T_r2_x, T_r2_y = testt.tester(encoder, decoder, T_test_loader)
        print("R2x : ", T_r2_x)
        print("R2y : ", T_r2_y)
        a = [[T_r2_x, T_r2_y, total_time]]
        if target == 2:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
            
elif param.mode == 'DAN':
    save_name = 'DANN'
    for target in range(2, 38):
        S_train_loader, S_val_loader = dataloader.Source_dataloader(target-1)
        T_train_loader, T_test_loader = dataloader.Target_dataloader(target)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            decoder = model.Decoder().cuda()
            discriminator = model.Discriminator().cuda()
            total_time = train.DANN_train(encoder, decoder, discriminator, 
                       S_train_loader, T_train_loader, save_name, param.target_label)
        else:
            print("There is no GPU -_-!")
        
        S_r2_x, S_r2_y, T_r2_x, T_r2_y, accuracy = testt.DA_tester(encoder, decoder, discriminator, S_val_loader, T_test_loader)
        print("R2x : ", T_r2_x)
        print("R2y : ", T_r2_y)
        a = [[S_r2_x, S_r2_y, T_r2_x, T_r2_y, accuracy, total_time]]
        if target == 2:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
            
              
elif param.mode == 'DDC':
    for target in range(2, 38):
        S_train_loader, S_val_loader = dataloader.New_Dataloader(target-1)
        T_train_loader, T_test_loader = dataloader.New_Dataloader(target)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            decoder = model.Decoder().cuda()
            total_time = train.DDN_train(encoder, decoder, S_train_loader, T_train_loader, param.target_label)
        else:
            print("There is no GPU -_-!")
            
        S_r2_x, S_r2_y, T_r2_x, T_r2_y, discrepancy = testt.DC_tester(encoder, decoder, S_val_loader, T_test_loader)
        print("R2x : ", T_r2_x)
        print("R2y : ", T_r2_y)
        a = [[S_r2_x, S_r2_y, T_r2_x, T_r2_y, discrepancy, total_time]]
        if target == 2:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
        
        
elif param.mode == 'MMD_tune':
    train_loader, val_loader = dataloader.New_Dataloader(1)
    
    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        decoder = model.Decoder().cuda()
        time = train.GRU_train(encoder, decoder, train_loader)
    else:
        print("There is no GPU -_-!")
        
    for target in range(2, 38):
        S_train_loader, S_val_loader = dataloader.New_Dataloader(target-1)
        T_train_loader, T_test_loader = dataloader.New_Dataloader(target)
        
        if torch.cuda.is_available():
            get_free_gpu()
            print('Running GPU : {}'.format(torch.cuda.current_device()))
            encoder = model.Extractor().cuda()
            encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
            decoder = model.Decoder().cuda()
            decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
            discrepancy = testt.embedding_MMD(encoder, decoder, S_train_loader, T_train_loader)
            print('\ndiscrepancy: ', discrepancy)
            if discrepancy < 0.5:
                total_time = train.GRU_train(encoder, decoder, T_train_loader)
            else:
                del encoder
                del decoder
                encoder = model.Extractor().cuda()
                decoder = model.Decoder().cuda()
                total_time = train.GRU_train(encoder, decoder, T_train_loader)
        else:
            print("There is no GPU -_-!")
        T_r2_x, T_r2_y = testt.tester(encoder, decoder, T_test_loader)
        print("R2x : ", T_r2_x)
        print("R2y : ", T_r2_y)
        a = [[T_r2_x, T_r2_y, total_time, discrepancy]]
        if target == 2:
            A = a
        else:
            A = np.concatenate((A, a), axis = 0)
             
        
# # fine-tune
# if __name__ == "__main__":
#     # pre-training
#     a = []
#     train_loader, val_loader = dataloader.New_Dataloader(1)
    
#     if torch.cuda.is_available():
#         get_free_gpu()
#         print('Running GPU : {}'.format(torch.cuda.current_device()))
#         encoder = model.Extractor().cuda()
#         decoder = model.Decoder().cuda()
#         time = train.GRU_train(encoder, decoder, train_loader)
#     else:
#         print("There is no GPU -_-!")
        
#     # r2_x, r2_y = testt.tester(encoder, decoder, val_loader)
    
#     for target in range(2, 38):
#         print('SESSION', target)
#         source_train, source_val = dataloader.New_Dataloader(target-1)
#         target_train, target_test = dataloader.New_Dataloader(target)
        
#         if torch.cuda.is_available():
#             get_free_gpu()
#             print('Running GPU : {}'.format(torch.cuda.current_device()))
#             encoder = model.Extractor().cuda()
#             encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
#             decoder = model.Decoder().cuda()
#             decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
#             # pretrained_dict = torch.load('pre_train_models/pre_discriminator.pt')
#             # model_dict = model.Discriminator().state_dict()
#             # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             # model_dict.update(pretrained_dict)
#             # discriminator.load_state_dict(model_dict)
            
#             total_time = train.DDN_train(encoder, decoder, source_train, target_train)
#         else:
#             print("There is no GPU -_-!")
        
#         T_r2_x, T_r2_y = testt.tester(encoder, decoder, target_test)
#         print("R2x : ", T_r2_x)
#         print("R2y : ", T_r2_y)
#         a = [[T_r2_x, T_r2_y, total_time]]
#         if target == 2:
#             A = a
#         else:
#             A = np.concatenate((A, a), axis = 0)


# fine-tune all
# A_x = np.zeros([37, 37])
# A_y = np.zeros([37, 37])
# if __name__ == "__main__":
#     for pre_train in range(37):
#         train_loader, val_loader = dataloader.New_Dataloader(pre_train+1)
        
#         if torch.cuda.is_available():
#             get_free_gpu()
#             print('Running GPU : {}'.format(torch.cuda.current_device()))
#             encoder = model.Extractor().cuda()
#             decoder = model.Decoder().cuda()
#             time = train.GRU_train(encoder, decoder, train_loader)
#         else:
#             print("There is no GPU -_-!")
            
#         r2_x, r2_y = testt.tester(encoder, decoder, val_loader)
        
#         for fine_tune in range(37):
#             if fine_tune > pre_train:
#                 T_tune_loader, T_test_loader = dataloader.New_Dataloader(fine_tune+1)
                
#                 if torch.cuda.is_available():
#                     get_free_gpu()
#                     print('Running GPU : {}'.format(torch.cuda.current_device()))
#                     encoder = model.Extractor().cuda()
#                     encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
#                     decoder = model.Decoder().cuda()
#                     decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
#                     # pretrained_dict = torch.load('pre_train_models/pre_discriminator.pt')
#                     # model_dict = model.Discriminator().state_dict()
#                     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#                     # model_dict.update(pretrained_dict)
#                     # discriminator.load_state_dict(model_dict)
                    
#                     total_time = train.GRU_train(encoder, decoder, T_tune_loader)
#                 else:
#                     print("There is no GPU -_-!")
                
#                 T_r2_x, T_r2_y = testt.tester(encoder, decoder, T_test_loader)
#                 if T_r2_x < 0:
#                     T_r2_x = 0
#                 if T_r2_y < 0:
#                     T_r2_y = 0
#             else:
#                 T_r2_x = 0
#                 T_r2_y = 0
            
#             A_x[-pre_train-1, fine_tune] = T_r2_x
#             A_y[-pre_train-1, fine_tune] = T_r2_y

# classes = ['0407_02', '0411_01', '0411_02', '0418_01', '0419_01', 
#             '0420_01', '0426_01', '0622_01', '0624_03', '0627_01', 
#             '0630_01', '0915_01', '0916_01', '0921_01', '0927_04', 
#             '0927_06', '0930_02', '0930_05', '1005_06', '1006_02', 
#             '1007_02', '1011_03', '1013_03', '1014_04', '1017_02', 
#             '1024_03', '1025_04', '1026_03', '1027_03', '1206_02', 
#             '1207_02', '1212_02', '1220_02', '0123_02', '0124_01', 
#             '0127_03', '0131_02']


# fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
# ax = sns.heatmap(A_x, vmin=0, vmax=0.9, square=True, xticklabels=classes, yticklabels=classes[::-1], cmap='plasma')
# ax.set_title('Fine-tuned Model Performance Heat Map (Vx)', fontsize = 30)
# ax.set_xlabel('Tuning Session',  fontsize = 24) 
# ax.set_ylabel('Training Session',  fontsize = 24) 
# fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
# ax = sns.heatmap(A_y, vmin=0, vmax=0.9, square=True, xticklabels=classes, yticklabels=classes[::-1], cmap='plasma')
# ax.set_title('Fine-tuned Model Performance Heat Map (Vy)', fontsize = 30)
# ax.set_xlabel('Tuning Session',  fontsize = 24) 
# ax.set_ylabel('Training Session',  fontsize = 24) 



# fine-tune cross
# if __name__ == "__main__":
#     a = []
#     for target in range(param.pre_train_num+1, 38):
#         print('SESSION', target)
#         train_loader, val_loader = dataloader.Multiple_Dataloader(target)
        
#         if torch.cuda.is_available():
#             get_free_gpu()
#             print('Running GPU : {}'.format(torch.cuda.current_device()))
#             encoder = model.Extractor().cuda()
#             decoder = model.Decoder().cuda()
#             time = train.GRU_train(encoder, decoder, train_loader, fine_tuning=False)
#         else:
#             print("There is no GPU -_-!")
            
#         # r2_x, r2_y = testt.tester(encoder, decoder, val_loader)
        
#         T_tune_loader, T_test_loader = dataloader.New_Dataloader(target)
        
#         if torch.cuda.is_available():
#             get_free_gpu()
#             print('Running GPU : {}'.format(torch.cuda.current_device()))
#             encoder = model.Extractor().cuda()
#             encoder.load_state_dict(torch.load('pre_train_models/encoder.pt'))
#             decoder = model.Decoder().cuda()
#             decoder.load_state_dict(torch.load('pre_train_models/decoder.pt'))
#             # pretrained_dict = torch.load('pre_train_models/pre_discriminator.pt')
#             # model_dict = model.Discriminator().state_dict()
#             # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             # model_dict.update(pretrained_dict)
#             # discriminator.load_state_dict(model_dict)
            
#             total_time = train.GRU_train(encoder, decoder, T_tune_loader, fine_tuning=True)
#         else:
#             print("There is no GPU -_-!")
        
#         T_r2_x, T_r2_y = testt.tester(encoder, decoder, T_test_loader)
#         print("R2x : ", T_r2_x)
#         print("R2y : ", T_r2_y)
#         a = [[T_r2_x, T_r2_y, total_time]]
#         if target == param.pre_train_num+1:
#             A = a
#         else:
#             A = np.concatenate((A, a), axis = 0)