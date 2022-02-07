#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:28:09 2020

@author: ogk
"""

import torch
import numpy as np
import pandas as pd
import utils
import torch.optim as optim
import torch.nn as nn
import testt
from utils import save_model
from utils import visualize
import param
import matplotlib.pyplot as plt
import os
import time
import MMD

def GRU_train(encoder, decoder, train_dataloader):
    print("Training")
    loss = []
    num_epoch = []
    time_start = time.time()
    for epoch in range(param.EPOCHS):
#         print('Epoch : {}'.format(epoch+1))
        encoder = encoder.train()
        decoder = decoder.train()
        
        decoder_criterion = nn.MSELoss().cuda()
        
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(decoder.parameters()),
            lr=0.001)
        
        for batch_idx, (data, movement) in enumerate(train_dataloader):
            
            data_input, data_mov_x, data_mov_y = data.cuda(), movement[:, -1:, 0].cuda(), movement[:, -1:, 1].cuda()
            
            optimizer = optimizer
            optimizer.zero_grad()
            
            feature = encoder(data_input)
            pred_x, pred_y = decoder(feature)
            
            decode_loss_x = decoder_criterion(pred_x, data_mov_x) # S_data_mov_x
            decode_loss_y = decoder_criterion(pred_y, data_mov_y) # S_data_mov_y
            
            total_loss = decode_loss_x + decode_loss_y
            total_loss.backward()
            optimizer.step()
            
#             if (batch_idx + 1) % 20 == 0:
#                 print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     batch_idx * len(data_input), 
#                     len(train_dataloader.dataset), 
#                     100. * batch_idx / len(train_dataloader), 
#                     total_loss.item()))
                
        loss.append(total_loss)
        num_epoch.append(epoch+1)
        
    time_end = time.time()
    total_time = time_end - time_start
    plt.plot(num_epoch, loss)
    
    save_folder = 'pre_train_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    torch.save(encoder.state_dict(), 'pre_train_models/encoder'+'.pt')
    torch.save(decoder.state_dict(), 'pre_train_models/decoder'+'.pt')
    
    return total_time


def DANN_train(encoder, decoder, discriminator, S_train_dataloader, T_train_dataloader, save_name, target_label):
    print("Domain adversarial training")
    loss = []
    num_epoch = []
    time_start = time.time()
    for epoch in range(param.EPOCHS):
#         print('Epoch : {}'.format(epoch+1))

        encoder = encoder.train()
        decoder = decoder.train()
        discriminator = discriminator.train()

        decoder_criterion = nn.MSELoss().cuda()
        discriminator_criterion = nn.bceLoss().cuda() # CrossEntropyLoss

        start_steps = epoch * len(S_train_dataloader)
        total_steps = param.EPOCHS * len(S_train_dataloader)
        
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(decoder.parameters()) +
            list(discriminator.parameters()),
            lr=0.0001)
        
        for batch_idx, (source_data, target_data) in enumerate(zip(S_train_dataloader, T_train_dataloader)):
            
            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            S_data, S_movement, S_label = source_data
            T_data, T_movement, T_label = target_data
            S_data_input, S_data_mov_x, S_data_mov_y, S_data_class = S_data.cuda(), S_movement[:, -1:, 0].cuda(), S_movement[:, -1:, 1].cuda(), S_label.cuda()
            T_data_input, T_data_mov_x, T_data_mov_y, T_data_class = T_data.cuda(), T_movement[:, -1:, 0].cuda(), T_movement[:, -1:, 1].cuda(), T_label.cuda()
            S_data_class = S_data_class.type(torch.long)
            T_data_class = T_data_class.type(torch.long)
            combined_input = torch.cat((S_data_input, T_data_input), 0)
            combined_class = torch.cat((S_data_class, T_data_class), 0)
            combined_vx = torch.cat((S_data_mov_x, T_data_mov_x), 0)
            combined_vy = torch.cat((S_data_mov_y, T_data_mov_y), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()
            
            combined_feature = encoder(combined_input)
            source_feature = encoder(S_data_input)
            
            # 1.Regress loss
            if target_label == True:
                label_pred_x, label_pred_y = decoder(combined_feature)
                decode_loss_x = decoder_criterion(label_pred_x, combined_vx)
                decode_loss_y = decoder_criterion(label_pred_y, combined_vy)
                # print('decode_loss_x', decode_loss_x)
                decode_loss = (decode_loss_x + decode_loss_y)
            else:
                label_pred_x, label_pred_y = decoder(source_feature)
                decode_loss_x = decoder_criterion(label_pred_x, S_data_mov_x)
                decode_loss_y = decoder_criterion(label_pred_y, S_data_mov_y)
                decode_loss = (decode_loss_x + decode_loss_y)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)
            domain_loss = discriminator_criterion(domain_pred, combined_class)

            total_loss = decode_loss + domain_loss
            total_loss.backward()
            optimizer.step()

#             if (batch_idx + 1) % 10 == 0:
#                 print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDecode Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
#                     batch_idx * len(S_data_input), len(S_train_dataloader.dataset), 100. * batch_idx / len(S_train_dataloader), 
#                     total_loss.item(), decode_loss.item(), domain_loss.item()))
                
        loss.append(total_loss.item())
        num_epoch.append(epoch+1)
        
    time_end = time.time()
    total_time = time_end - time_start
    
    plt.plot(num_epoch, loss)

    save_model(encoder, decoder, discriminator, 'training', save_name)
    
    return total_time
    
    
def DDN_train(encoder, decoder, S_train_dataloader, T_train_dataloader, target_label):
    print('Deep domain confusion training')
    loss = []
    num_epoch = []
    time_start = time.time()
    for epoch in range(param.EPOCHS):
#         print('Epoch : {}'.format(epoch+1))
        encoder = encoder.train()
        decoder = decoder.train()
        
        decoder_criterion = nn.MSELoss().cuda()
        
        optimizer = optim.Adam(
            list(encoder.parameters()) +
            list(decoder.parameters()),
            lr=0.001)
        
        for batch_idx, (source_data, target_data) in enumerate(zip(S_train_dataloader, T_train_dataloader)):
            
            S_data, S_movement = source_data
            T_data, T_movement = target_data
            S_data_input, S_data_mov_x, S_data_mov_y = S_data.cuda(), S_movement[:, -1:, 0].cuda(), S_movement[:, -1:, 1].cuda()
            T_data_input, T_data_mov_x, T_data_mov_y = T_data.cuda(), T_movement[:, -1:, 0].cuda(), T_movement[:, -1:, 1].cuda()            
            
            if S_data_input.size(0)!=param.batch_size:
                break
            
            combined_input = torch.cat((S_data_input, T_data_input), 0)
            combined_vx = torch.cat((S_data_mov_x, T_data_mov_x), 0)
            combined_vy = torch.cat((S_data_mov_y, T_data_mov_y), 0)
            
            optimizer = optimizer
            optimizer.zero_grad()
            
            combined_feature = encoder(combined_input)
            source_feature = encoder(S_data_input)
            target_feature = encoder(T_data_input)
            
            # 1.Regress loss
            if target_label == True:
                label_pred_x, label_pred_y = decoder(combined_feature)
                decode_loss_x = decoder_criterion(label_pred_x, combined_vx)
                decode_loss_y = decoder_criterion(label_pred_y, combined_vy)
                decode_loss = (decode_loss_x + decode_loss_y)
            else:
                label_pred_x, label_pred_y = decoder(source_feature)
                decode_loss_x = decoder_criterion(label_pred_x, S_data_mov_x)
                decode_loss_y = decoder_criterion(label_pred_y, S_data_mov_y)
                decode_loss = (decode_loss_x + decode_loss_y)
                
            # 2. Domain loss
            domain_loss = MMD.mmd(source_feature, target_feature)
            
            total_loss = decode_loss + 5*domain_loss
            total_loss.backward()
            optimizer.step()
            
#             if (batch_idx + 1) % 50 == 0:
#                 print('[{}/{} ({:.0f}%)]\ttotal Loss: {:.6f}\tdecode loss:{:.6f}\tdomain loss:{:.6f}'.format(
#                     batch_idx * len(S_data_input), len(S_train_dataloader.dataset), 100. * batch_idx / len(S_train_dataloader), 
#                     total_loss.item(), decode_loss.item(), domain_loss.item()))
                
        loss.append(total_loss.item())
        num_epoch.append(epoch+1)
        
    time_end = time.time()
    total_time = time_end - time_start
    
    plt.plot(num_epoch, loss)
    
    save_folder = 'pre_train_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    torch.save(encoder.state_dict(), 'pre_train_models/encoder'+'.pt')
    torch.save(decoder.state_dict(), 'pre_train_models/decoder'+'.pt')
    
    return total_time

        

