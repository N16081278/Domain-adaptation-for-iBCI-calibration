import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import itertools
from dataPreprocess import read_dataset
from model_tf import DomainAdversarialNetwork

TRAIN_COUNT = 5000

def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

for n in range(5):
    for sourceSessionIndex, targetSessionIndex in itertools.product(range(37), range(37)):

        if sourceSessionIndex == targetSessionIndex:
            continue

        folderPath = '../BCI_movementDecoder/data/indy'
        fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

        sourceSession = fileList[sourceSessionIndex]
        targetSession = fileList[targetSessionIndex]

        sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
        targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    
        print('count =>', n, 'sourceIndex =>', sourceSessionIndex, 'targetIndex =>', targetSessionIndex)

        # model
        model = DomainAdversarialNetwork()
        model.compile(optimizer = 'adam', run_eagerly=True)

        # prepare dataset     

        train_x = {
            'source': sourceDataset['m1'][:TRAIN_COUNT], 
            'target': targetDataset['m1'][:TRAIN_COUNT]
            }
        train_y = {
            'source_movement': sourceDataset['movement'][:TRAIN_COUNT], 
            'target_movement': targetDataset['movement'][:TRAIN_COUNT], 
            'source_domain': tf.one_hot(np.zeros([TRAIN_COUNT]), 2), 
            'target_domain': tf.one_hot(np.ones([TRAIN_COUNT]), 2)
        }

        test_x = {
            'source': np.zeros_like(targetDataset['m1'][TRAIN_COUNT:]), 
            'target': targetDataset['m1'][TRAIN_COUNT:]
        }

        test_y = targetDataset['movement'][TRAIN_COUNT:]

 
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128)

        
        # train
        model.useTargetLabel = True
        epochs = 100
        for epoch in range(epochs):
            start_steps = epoch * len(train_dataset)
            total_steps = epochs * len(train_dataset)

            logs = []

            for step, (x, y) in enumerate(train_dataset):
            
                p = float(step + start_steps) / total_steps
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                model.alpha = alpha       
                log = model.train_on_batch(x=x, y=y, return_dict=True)
                logs.append(log)
            
            df = pd.DataFrame(logs)
            print('epoch', epoch, 'decode_loss', df.mean()['decode_loss'], 'domain_loss', df.mean()['domain_loss'])

        # test
        model.predict_movement = True
        pred = model.predict(x=test_x)


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
            'model': ['DomainAdversarial']
        })
        
        result = result.explode(['axis', 'r_square']).reset_index(drop=True)

        result.to_csv('./dan_study.csv', index=False, header=False, mode='a')


