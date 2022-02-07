import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import itertools
from dataPreprocess import read_dataset
from model_tf import simpleDecodeModel

def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = np.concatenate([source, target], axis=0)
   
    bs_cnt = int(total.shape[0])
    feature_cnt = int(total.shape[1])

    total0 = np.broadcast_to(np.expand_dims(total, axis=0), shape=[bs_cnt, bs_cnt, feature_cnt])
    total1 = np.broadcast_to(np.expand_dims(total, axis=1), shape=[bs_cnt, bs_cnt, feature_cnt])
    L2_distance = np.sum((total0-total1)**2, axis=2) 

 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = np.sum(L2_distance) / (n_samples**2-n_samples)
   
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

   
    kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)

def maximumMeanDiscrepancy(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.shape[0])
    m = int(target.shape[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
   
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    
    XX = np.reshape(np.sum(np.divide(XX, n * n), axis=1), [1, -1])  # K_ss矩阵，Source<->Source
    XY = np.reshape(np.sum(np.divide(XY, -n * m), axis=1), [1, -1]) # K_st矩阵，Source<->Target
    YX = np.reshape(np.sum(np.divide(YX, -m * n), axis=1), [1, -1]) # K_ts矩阵,Target<->Source
    YY = np.reshape(np.sum(np.divide(YY, m * m), axis=1), [1, -1]) # K_tt矩阵,Target<->Target

    	
    loss = np.sum(XX + XY) + np.sum(YX + YY)
    
    return loss

for n in range(5):
    for sourceSessionIndex, targetSessionIndex in itertools.product(range(37), range(37)):

        if sourceSessionIndex != targetSessionIndex:
            continue

        folderPath = '../BCI_movementDecoder/data/indy'
        fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

        sourceSession = fileList[sourceSessionIndex]
        targetSession = fileList[targetSessionIndex]

        sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
        targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    
        print('count =>', n, 'sourceIndex =>', sourceSessionIndex, 'targetIndex =>', targetSessionIndex)

        # model
        model = simpleDecodeModel()
        model.compile(optimizer = 'adam', run_eagerly=True, loss='mse')

        # prepare dataset
        TRAIN_COUNT = 5000
        train_x = sourceDataset['m1'][:TRAIN_COUNT]
        train_y = sourceDataset['movement'][:TRAIN_COUNT]

        test_x = sourceDataset['m1'][TRAIN_COUNT:]
        test_y = sourceDataset['movement'][TRAIN_COUNT:]

        # fit
        model.fit(x=sourceDataset['m1'][:TRAIN_COUNT], y=sourceDataset['movement'][:TRAIN_COUNT], \
            batch_size=128, epochs=100, shuffle=True, verbose=1)
        
        # pred
        model.return_feature = False
        source_pred = model.predict(x=sourceDataset['m1'][TRAIN_COUNT:])
        source_true = sourceDataset['movement'][TRAIN_COUNT:]
        source_r2 = [r_square(pred=source_pred[:, i], true=source_true[:, i]) for i in range(source_pred.shape[-1])]

        target_pred = model.predict(x=targetDataset['m1'][TRAIN_COUNT:])
        target_true = targetDataset['movement'][TRAIN_COUNT:]
        target_r2 = [r_square(pred=target_pred[:, i], true=target_true[:, i]) for i in range(target_pred.shape[-1])]

        model.return_feature = True
        mmd_loss = 0.0
        for i in range(0, TRAIN_COUNT, 1000):
            _, source_feature = model.predict(x=sourceDataset['m1'][i:i+1000])
            _, target_feature = model.predict(x=targetDataset['m1'][i:i+1000])
            mmd_loss += maximumMeanDiscrepancy(source_feature, target_feature)
        mmd_loss /= 5

        # save
        result = pd.DataFrame({
            'source_r_square': [source_r2],
            'target_r_square': [target_r2],
            'axis': [['x', 'y']],
            'movement': 'velocity',
            'sourceSessionIndex': [sourceSessionIndex],
            'targetSessionIndex': [targetSessionIndex],
            'normalizedMethod': ['None'],
            'maximumMeanDiscrepancy': [mmd_loss]
        })
        
        result = result.explode(['axis', 'source_r_square', 'target_r_square']).reset_index(drop=True)

        result.to_csv('./mmd_study.csv', index=False, header=False, mode='a')


