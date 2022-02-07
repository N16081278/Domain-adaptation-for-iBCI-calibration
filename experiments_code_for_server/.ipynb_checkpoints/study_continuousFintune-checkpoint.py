import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os

from tensorflow.python.keras.models import Model
from dataPreprocess import read_dataset
from model_tf import simpleDecodeModel

def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

folderPath = '../BCI_movementDecoder/data/indy'
fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

for n in range(1):
    for sourceSessionIndex in range(36):
        targetSessionIndex = sourceSessionIndex + 1    
        
        sourceSession = fileList[sourceSessionIndex]
        sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])

        targetSession = fileList[targetSessionIndex]
        targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

        ################################################################
        # RETRAIN        
        model = simpleDecodeModel()
        model.compile(loss='mse', optimizer='adam', run_eagerly=True)
        
        train_x = sourceDataset['m1'][:5000]
        train_y = sourceDataset['movement'][:5000]
        model.fit(x=train_x, y=train_y, batch_size=256, epochs=100, verbose=1, shuffle=True)
     
        model.save_weights(f'./model/non_{sourceSessionIndex}')

        if sourceSessionIndex == 0:
            model.save_weights(f'./model/continuous_{sourceSessionIndex}')
        del model
        ##############################################################
        # FINETUNE - NON
        model_non = simpleDecodeModel()
        model_non(tf.zeros([1, 30, 96]))
        model_non.load_weights(f'./non_{sourceSessionIndex}')
        model_non.compile(loss='mse', optimizer='adam', run_eagerly=True)
        
        train_x = targetDataset['m1'][:5000]
        train_y = targetDataset['movement'][:5000]

        model_non.fit(x=train_x, y=train_y, batch_size=256, epochs=100, verbose=1, shuffle=True)
        
        test_x = targetDataset['m1'][5000:]
        test_y = targetDataset['movement'][5000:]
        pred = model_non.predict(test_x)

        r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]
        
        df = pd.DataFrame({
            'r_square': [r2],
            'axis': [['x', 'y']],
            'movement': ['velocity'],
            'sourceSessionIndex': [sourceSessionIndex],
            'targetSessionIndex': [targetSessionIndex],
            'normalizedMethod': ['None'],
            'model': ['Finetune'],            
        })
        df = df.explode(['axis', 'r_square']).reset_index(drop=True)
        df.to_csv('./continuousFinetuneTest.csv', index=False, header=False, mode='a')
        del model_non
        ##############################################################
        # FINETUNE - CONTINUOUS
        model_continuous = simpleDecodeModel()
        model_continuous(tf.zeros([1, 30, 96]))
        model_continuous.load_weights(f'./model/continuous_{sourceSessionIndex}')
        model_continuous.compile(loss='mse', optimizer='adam', run_eagerly=True)

        train_x = targetDataset['m1'][:5000]
        train_y = targetDataset['movement'][:5000]

        model_continuous.fit(x=train_x, y=train_y, batch_size=256, epochs=100, verbose=1, shuffle=True)

        test_x = targetDataset['m1'][5000:]
        test_y = targetDataset['movement'][5000:]
        pred = model_continuous.predict(test_x)

        r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]
        
        df = pd.DataFrame({
            'r_square': [r2],
            'axis': [['x', 'y']],
            'movement': ['velocity'],
            'sourceSessionIndex': [sourceSessionIndex],
            'targetSessionIndex': [targetSessionIndex],
            'normalizedMethod': ['None'],
            'model': ['ContinuousFinetune'],            
        })
        df = df.explode(['axis', 'r_square']).reset_index(drop=True)
        df.to_csv('./continuousFinetuneTest.csv', index=False, header=False, mode='a')

        model_continuous.save(f'./model/continuous_{targetSessionIndex}')
        del model_continuous