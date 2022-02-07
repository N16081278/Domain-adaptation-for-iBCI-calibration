import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import itertools
from dataPreprocess import read_dataset
from model_tf import simpleDecodeModel

for sourceSessionIndex, targetSessionIndex in itertools.product(range(37), range(37)):

    if sourceSessionIndex + 1 != targetSessionIndex:
        continue

    folderPath = '../BCI_movementDecoder/data/indy'
    fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

    sourceSession = fileList[sourceSessionIndex]
    targetSession = fileList[targetSessionIndex]

    sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
    targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    for n in range(5):
        print('count =>', n, 'sourceIndex =>', sourceSessionIndex, 'targetIndex =>', targetSessionIndex)

        # model
        model = simpleDecodeModel()
        model.compile(optimizer = 'adam', run_eagerly=True, loss='mse')

        # prepare dataset
        TRAIN_COUNT = 5000
        train_x = np.concatenate((sourceDataset['m1'][:TRAIN_COUNT], targetDataset['m1'][:TRAIN_COUNT]), axis=0)
        train_y = np.concatenate((sourceDataset['movement'][:TRAIN_COUNT], targetDataset['movement'][:TRAIN_COUNT]), axis=0)

        test_x = targetDataset['m1'][TRAIN_COUNT:]
        test_y = targetDataset['movement'][TRAIN_COUNT:]
  
        # z score
        # t_mean, t_std = train_y['source_movement'].mean(), train_y['target_movement'].std()

        # train
        # model.useTargetLabel = True
        model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=2, shuffle=True)

        # test
        # model.predict_movement = True
        pred = model.predict(x=test_x)

        # r2
        def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
            ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
            ss_res = np.sum((true - pred) ** 2, axis=0)
            r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

            return r_square

        # 計算R2
        r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]

        # save
        result = pd.DataFrame({
            'r_square': [r2],
            'axis': [['x', 'y']],
            'movement': 'velocity',
            'sourceSessionIndex': [sourceSessionIndex],
            'targetSessionIndex': [targetSessionIndex],
            'normalizedMethod': ['None'],
            'model': ['MixCalibration']
        })
        
        result = result.explode(['axis', 'r_square']).reset_index(drop=True)

        result.to_csv('./mix_study.csv', index=False, header=False, mode='a')


