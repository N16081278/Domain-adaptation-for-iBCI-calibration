#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:52:46 2020

@author: ogk
"""


import torch
import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # confusion matrix
from function import plot_confusion_matrix # CM plot
from sklearn import manifold # t-SNE
from sklearn.metrics import r2_score
import MMD


def tester(encoder, decoder, T_test_dataloader):
# def tester(encoder, decoder, discriminator, test_dataloader, training_mode):
    print("Model test ...")

    encoder.cuda()
    decoder.cuda()

    all_preds = torch.tensor([]).cuda()
    all_label = torch.tensor([]).cuda()
    
    for batch_idx, (data, movement) in enumerate(T_test_dataloader):
        
        data_input, data_mov_x, data_mov_y = data.cuda(), movement[:, -1, 0].cuda(), movement[:, -1, 1].cuda()
        
        feature = encoder(data_input)
        label_pred_x, label_pred_y = decoder(feature)
        out_x = label_pred_x.view(-1,1)
        out_y = label_pred_y.view(-1,1)
        
        if batch_idx == 0:
            pred_x = out_x.data.cpu().numpy()
            pred_y = out_y.data.cpu().numpy()
            data_motion_x = data_mov_x.cpu().numpy()
            data_motion_y = data_mov_y.cpu().numpy()
            embedding = feature.cpu().detach().numpy()
        else:
            pred_x = np.concatenate((pred_x, out_x.data.cpu().numpy()), axis = 0)
            pred_y = np.concatenate((pred_y, out_y.data.cpu().numpy()), axis = 0)
            data_motion_x = np.concatenate((data_motion_x, data_mov_x.cpu().numpy()), axis = 0)
            data_motion_y = np.concatenate((data_motion_y, data_mov_y.cpu().numpy()), axis = 0)
            embedding = np.concatenate((embedding, feature.detach().cpu().numpy()), axis = 0)
            
    
    r2_x = r2_score(data_motion_x, pred_x)
    r2_y = r2_score(data_motion_y, pred_y)
        
    
    
    
    # # Confusion matrix
    
    # print('\nploting the confusion matrix')
    # C = confusion_matrix(all_label.cpu(), all_preds.cpu(), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # classes = ['0407_02', '0411_01', '0411_02', '0418_01', '0419_01', 
    #         '0420_01', '0426_01', '0622_01', '0624_03', '0627_01', 
    #         '0630_01', '0915_01', '0916_01', '0921_01', '0927_04', 
    #         '0927_06', '0930_02', '0930_05', '1005_06', '1006_02', 
    #         '1007_02', '1011_03', '1013_03', '1014_04', '1017_02', 
    #         '1024_03', '1025_04', '1026_03', '1027_03', '1206_02', 
    #         '1207_02', '1212_02', '1220_02', '0123_02', '0124_01', 
    #         '0127_03', '0131_02']
    # plt.figure()
    # plot_confusion_matrix(C, classes[:param.num_session], normalize=True, title='confusion matrix')
    # plt.show()
    # cm = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    
    
    #t-SNE
    
    # print('\nt-SNE embedding visualizing')
    # X_tsne = manifold.TSNE(n_components=18, init='random', random_state=5, verbose=1).fit_transform(embedding)
    # plt.figure(figsize=(10, 10))
    # df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=labelagan.squeeze()))
    # df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    return r2_x, r2_y


def DA_tester(encoder, decoder, discriminator, S_test_dataloader, T_test_dataloader):
    print("Model test ...")

    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    domain_correct = 0

    all_preds = torch.tensor([]).cuda()
    all_label = torch.tensor([]).cuda()
    for batch_idx, (source_data, target_data) in enumerate(zip(S_test_dataloader, T_test_dataloader)): # (data, movement, label)
        p = float(batch_idx) / len(S_test_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # 1. Source input -> Source Regression
        S_data, S_movement, S_label = source_data
        T_data, T_movement, T_label = target_data
        S_data_input, S_data_mov_x, S_data_mov_y, S_data_class = S_data.cuda(), S_movement[:, -1, 0].cuda(), S_movement[:, -1, 1].cuda(), S_label.cuda()
        T_data_input, T_data_mov_x, T_data_mov_y, T_data_class = T_data.cuda(), T_movement[:, -1, 0].cuda(), T_movement[:, -1, 1].cuda(), T_label.cuda()
        source_feature = encoder(S_data_input)
        target_feature = encoder(T_data_input)
        S_label_pred_x, S_label_pred_y = decoder(source_feature)
        T_label_pred_x, T_label_pred_y = decoder(target_feature)
        S_out_x = S_label_pred_x.view(-1,1)
        S_out_y = S_label_pred_y.view(-1,1)
        T_out_x = T_label_pred_x.view(-1,1)
        T_out_y = T_label_pred_y.view(-1,1)
        
        if batch_idx == 0:
            S_pred_x = S_out_x.data.cpu().numpy()
            S_pred_y = S_out_y.data.cpu().numpy()
            S_data_motion_x = S_data_mov_x.cpu().numpy()
            S_data_motion_y = S_data_mov_y.cpu().numpy()
            T_pred_x = T_out_x.data.cpu().numpy()
            T_pred_y = T_out_y.data.cpu().numpy()
            T_data_motion_x = T_data_mov_x.cpu().numpy()
            T_data_motion_y = T_data_mov_y.cpu().numpy()
        else:
            S_pred_x = np.concatenate((S_pred_x, S_out_x.data.cpu().numpy()), axis = 0)
            S_pred_y = np.concatenate((S_pred_y, S_out_y.data.cpu().numpy()), axis = 0)
            S_data_motion_x = np.concatenate((S_data_motion_x, S_data_mov_x.cpu().numpy()), axis = 0)
            S_data_motion_y = np.concatenate((S_data_motion_y, S_data_mov_y.cpu().numpy()), axis = 0)
            T_pred_x = np.concatenate((T_pred_x, T_out_x.data.cpu().numpy()), axis = 0)
            T_pred_y = np.concatenate((T_pred_y, T_out_y.data.cpu().numpy()), axis = 0)
            T_data_motion_x = np.concatenate((T_data_motion_x, T_data_mov_x.cpu().numpy()), axis = 0)
            T_data_motion_y = np.concatenate((T_data_motion_y, T_data_mov_y.cpu().numpy()), axis = 0)
        
        # 3. Combined input -> Domain Classificaion
        combined_input = torch.cat((S_data_input, T_data_input), 0)
        combined_class = torch.cat((S_data_class, T_data_class), 0)
        domain_feature = encoder(combined_input)
        domain_output = discriminator(domain_feature, alpha)
        # data_class = combined_class.data.max(1, keepdim=True)[1]
        domain_pred = domain_output.data.max(1, keepdim=True)[1]
        all_preds = torch.cat((all_preds, domain_pred), dim=0)
        all_label = torch.cat((all_label, combined_class), dim=0)
        domain_correct += np.sum(combined_class.cpu().numpy().flatten() == domain_pred.cpu().numpy().flatten())
        if batch_idx == 0:
            embedding = domain_feature.cpu().detach().numpy()
            labelagan = combined_class.cpu().numpy()
        else:
            embedding = np.concatenate((embedding, domain_feature.detach().cpu().numpy()), axis = 0)
            labelagan = np.concatenate((labelagan, combined_class.cpu().numpy()), axis = 0)
    print('embedding.shape: ', embedding.shape)
    print('labelagan: ', labelagan.shape)
    
    S_r2_x = r2_score(S_data_motion_x, S_pred_x)
    S_r2_y = r2_score(S_data_motion_y, S_pred_y)
    T_r2_x = r2_score(T_data_motion_x, T_pred_x)
    T_r2_y = r2_score(T_data_motion_y, T_pred_y)
    accuracy = 100. * domain_correct.item() / (len(S_test_dataloader.dataset)+len(T_test_dataloader.dataset))
    
    # Confusion matrix
#     print('\nploting the confusion matrix')
#     C = confusion_matrix(all_label.cpu(), all_preds.cpu())
#     classes = ['source', 'target']
#     plt.figure()
#     plot_confusion_matrix(C, classes, normalize=True, title='confusion matrix')
#     plt.show()
    
    # cm = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    
    
    #t-SNE
    # print('\nt-SNE embedding visualizing')
    # X_tsne = manifold.TSNE(n_components=18, init='random', random_state=5, verbose=1).fit_transform(embedding)
    # plt.figure(figsize=(10, 10))
    # df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=labelagan.squeeze()))
    # df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    return S_r2_x, S_r2_y, T_r2_x, T_r2_y, accuracy


def DC_tester(encoder, decoder, S_test_dataloader, T_test_dataloader):
    print("Model test ...")

    encoder.cuda()
    decoder.cuda()

    all_preds = torch.tensor([]).cuda()
    all_label = torch.tensor([]).cuda()
    for batch_idx, (source_data, target_data) in enumerate(zip(S_test_dataloader, T_test_dataloader)): # (data, movement, label)
        
        # 1. Source input -> Source Regression
        S_data, S_movement = source_data
        T_data, T_movement = target_data
        S_data_input, S_data_mov_x, S_data_mov_y = S_data.cuda(), S_movement[:, -1, 0].cuda(), S_movement[:, -1, 1].cuda()
        T_data_input, T_data_mov_x, T_data_mov_y = T_data.cuda(), T_movement[:, -1, 0].cuda(), T_movement[:, -1, 1].cuda()
        source_feature = encoder(S_data_input)
        target_feature = encoder(T_data_input)
        S_label_pred_x, S_label_pred_y = decoder(source_feature)
        T_label_pred_x, T_label_pred_y = decoder(target_feature)
        S_out_x = S_label_pred_x.view(-1,1)
        S_out_y = S_label_pred_y.view(-1,1)
        T_out_x = T_label_pred_x.view(-1,1)
        T_out_y = T_label_pred_y.view(-1,1)
        
        # 2. Domain loss
        domain_loss = MMD.mmd(source_feature, target_feature)
        
        if batch_idx == 0:
            S_pred_x = S_out_x.data.cpu().numpy()
            S_pred_y = S_out_y.data.cpu().numpy()
            S_data_motion_x = S_data_mov_x.cpu().numpy()
            S_data_motion_y = S_data_mov_y.cpu().numpy()
            T_pred_x = T_out_x.data.cpu().numpy()
            T_pred_y = T_out_y.data.cpu().numpy()
            T_data_motion_x = T_data_mov_x.cpu().numpy()
            T_data_motion_y = T_data_mov_y.cpu().numpy()
            D = [domain_loss.cpu().data.numpy()]
        else:
            S_pred_x = np.concatenate((S_pred_x, S_out_x.data.cpu().numpy()), axis = 0)
            S_pred_y = np.concatenate((S_pred_y, S_out_y.data.cpu().numpy()), axis = 0)
            S_data_motion_x = np.concatenate((S_data_motion_x, S_data_mov_x.cpu().numpy()), axis = 0)
            S_data_motion_y = np.concatenate((S_data_motion_y, S_data_mov_y.cpu().numpy()), axis = 0)
            T_pred_x = np.concatenate((T_pred_x, T_out_x.data.cpu().numpy()), axis = 0)
            T_pred_y = np.concatenate((T_pred_y, T_out_y.data.cpu().numpy()), axis = 0)
            T_data_motion_x = np.concatenate((T_data_motion_x, T_data_mov_x.cpu().numpy()), axis = 0)
            T_data_motion_y = np.concatenate((T_data_motion_y, T_data_mov_y.cpu().numpy()), axis = 0)
            D = np.concatenate((D, [domain_loss.cpu().data.numpy()]), axis = 0)
        
        
        
    S_r2_x = r2_score(S_data_motion_x, S_pred_x)
    S_r2_y = r2_score(S_data_motion_y, S_pred_y)
    T_r2_x = r2_score(T_data_motion_x, T_pred_x)
    T_r2_y = r2_score(T_data_motion_y, T_pred_y)
    discrepancy = np.mean(D)
    
    #t-SNE
    # print('\nt-SNE embedding visualizing')
    # X_tsne = manifold.TSNE(n_components=18, init='random', random_state=5, verbose=1).fit_transform(embedding)
    # plt.figure(figsize=(10, 10))
    # df = pd.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=labelagan.squeeze()))
    # df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    
    return S_r2_x, S_r2_y, T_r2_x, T_r2_y, discrepancy


def embedding_MMD(encoder, decoder, S_test_dataloader, T_test_dataloader):
    print('embedding_MMD calculating...')
    
    encoder.cuda()
    decoder.cuda()
    all_preds = torch.tensor([]).cuda()
    all_label = torch.tensor([]).cuda()
    D = []
    for batch_idx, (source_data, target_data) in enumerate(zip(S_test_dataloader, T_test_dataloader)):
        S_data, S_movement = source_data
        T_data, T_movement = target_data
        S_data_input = S_data.cuda()
        T_data_input = T_data.cuda()
        source_feature = encoder(S_data_input)
        target_feature = encoder(T_data_input)
        
        # domain_loss = MMD.mmd(source_feature, target_feature)
        if batch_idx%3 == 0:
            s_embedding = source_feature
            t_embedding = target_feature
            # D = [domain_loss.cpu().data.numpy()]
        else:
            s_embedding = torch.cat((s_embedding, source_feature), 0)
            t_embedding = torch.cat((t_embedding, target_feature), 0)
            # D = np.concatenate((D, [domain_loss.cpu().data.numpy()]), axis = 0)
        if (batch_idx+1)%3 == 0:
            domain_loss = MMD.mmd(s_embedding, t_embedding)
            D = np.concatenate((D, [domain_loss.cpu().data.numpy()]), axis = 0)
    
    discrepancy = np.mean(D)
    
    return discrepancy



def C_tester(encoder, pre_discriminator, S_test_dataloader):
    print("classifier test ...")
    
    encoder.cuda()
    pre_discriminator.cuda()
    
    all_preds = torch.tensor([]).cuda()
    all_label = torch.tensor([]).cuda()
    correct = 0
    for batch_idx, (data, movement, label) in enumerate(S_test_dataloader):
        data_input, data_class = data.cuda(), label.cuda()
        feature = encoder(data_input)
        label = pre_discriminator(feature)
        data_class = data_class.data.max(1, keepdim=True)[1]
        label_pred = label.data.max(1, keepdim=True)[1]
        all_preds = torch.cat((all_preds, label_pred),dim=0)
        all_label = torch.cat((all_label, data_class),dim=0)
        correct += sum(data_class==label_pred)
        
        if batch_idx == 0:
            labelagan = data_class.cpu().numpy()
        else:
            labelagan = np.concatenate((labelagan, data_class.cpu().numpy()), axis = 0)
            
    accuracy = 100. * correct.item() / (len(S_test_dataloader.dataset))
    print('Accuracy: {}/{} ({:.2f}%)\n'.
            format(correct, len(S_test_dataloader.dataset), accuracy))
    
    print('ploting the confusion matrix')
    C = confusion_matrix(all_label.cpu(), all_preds.cpu())
    classes = ['0407_02', '0411_01', '0411_02', '0418_01', '0419_01', 
           '0420_01', '0426_01', '0622_01', '0624_03', '0627_01', 
           '0630_01', '0915_01', '0916_01', '0921_01', '0927_04', 
           '0927_06', '0930_02', '0930_05', '1005_06', '1006_02', 
           '1007_02', '1011_03', '1013_03', '1014_04', '1017_02', 
           '1024_03', '1025_04', '1026_03', '1027_03', '1206_02', 
           '1207_02', '1212_02', '1220_02', '0123_02', '0124_01', 
           '0127_03', '0131_02']
    plt.figure()
    plot_confusion_matrix(C, classes, normalize=True, title='confusion matrix')
    plt.show()
    
    return accuracy





    # return r2_x, r2_y, accuracy, embedding, labelagan