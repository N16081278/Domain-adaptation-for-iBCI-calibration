import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import itertools
from dataPreprocess import read_dataset
from model_tf import simpleDecodeModel, DeepDomainConfusionModel, DomainAdversarialNetwork

def r_square(pred:np.ndarray, true:np.ndarray) -> np.ndarray:
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((true - pred) ** 2, axis=0)
    r_square = np.ones_like(ss_tot) - (ss_res / ss_tot)

    return r_square

for n in range(1):
    for sourceSessionIndex, targetSessionIndex in itertools.product(range(37), range(37)):
        
        if not sourceSessionIndex + 1 == targetSessionIndex:
            continue
        if sourceSessionIndex < 32:
            continue
      
        folderPath = '../BCI_movementDecoder/data/indy'
        fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

        sourceSession = fileList[sourceSessionIndex]
        targetSession = fileList[targetSessionIndex]

        sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
        targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    
        print('count =>', n, 'sourceIndex =>', sourceSessionIndex, 'targetIndex =>', targetSessionIndex)

        sourceLength = 5000
        for targetLength in [938, 1875, 2813, 3750, 4688]:        
            print(f'[fintune] targetLength = {targetLength}')
            ###########################################################################
            # FINETUNE
            model = simpleDecodeModel()
            model.compile(loss='mse', optimizer='adam', run_eagerly=True)

            train_x = sourceDataset['m1'][:sourceLength]
            train_y = sourceDataset['movement'][:sourceLength]
            model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=0, shuffle=True)
       
            train_x = targetDataset['m1'][:targetLength]
            train_y = targetDataset['movement'][:targetLength]
            model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=0, shuffle=True)

            test_x = targetDataset['m1'][5000:]
            test_y = targetDataset['movement'][5000:]
            pred = model.predict(test_x)

            r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]

            df = pd.DataFrame({
                'r_square': [r2],
                'axis': [['x', 'y']],
                'movement': 'velocity',
                'sourceSessionIndex': [sourceSessionIndex],
                'targetSessionIndex': [targetSessionIndex],
                'normalizedMethod': ['None'],
                'model': ['Finetune'],
                'targetLength': targetLength
            })
            df = df.explode(['axis', 'r_square']).reset_index(drop=True)
            df.to_csv('./result_experiments/timeLengthTest.csv', index=False, header=False, mode='a')
            
            print(f'[mixCalibration] targetLength = {targetLength}')
            ###########################################################################
            # MIX
            model = simpleDecodeModel()
            model.compile(loss='mse', optimizer='adam', run_eagerly=True)

            train_x = sourceDataset['m1'][:sourceLength] 
            train_y = sourceDataset['movement'][:sourceLength]
            model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=0, shuffle=True)

            test_x = targetDataset['m1'][5000:]
            test_y = targetDataset['movement'][5000:]
            pred = model.predict(test_x)

            r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]

            df = pd.DataFrame({
                'r_square': [r2],
                'axis': [['x', 'y']],
                'movement': 'velocity',
                'sourceSessionIndex': [sourceSessionIndex],
                'targetSessionIndex': [targetSessionIndex],
                'normalizedMethod': ['None'],
                'model': ['MixCalibration'],
                'targetLength': targetLength
            })
            df = df.explode(['axis', 'r_square']).reset_index(drop=True)
            df.to_csv('./result_experiments/timeLengthTest.csv', index=False, header=False, mode='a')

            
            print(f'[domainComfusion] targetLength = {targetLength}')
            ###########################################################################
            # DeepDomainComfusion
            model = DeepDomainConfusionModel()
            model.compile(optimizer='adam', run_eagerly=True)
            
            padding_width_m1 = [[0, sourceLength-targetLength], [0, 0], [0, 0]]
            train_x = {
                'source': sourceDataset['m1'][:sourceLength], 
                'target': np.pad(targetDataset['m1'][:targetLength], padding_width_m1, 'wrap') 
                }
            padding_width_movement = [[0, sourceLength-targetLength], [0, 0]]
            train_y = {
                'source_movement': sourceDataset['movement'][:sourceLength], 
                'target_movement': np.pad(targetDataset['movement'][:targetLength], padding_width_movement, 'wrap'),                 
                }

            test_x = {
                'source': np.zeros_like(targetDataset['m1'][5000:]), 
                'target': targetDataset['m1'][5000:]
                }
            test_y = targetDataset['movement'][5000:]
            
            model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=0, shuffle=True)

            model.predict_movement = True
            pred = model.predict(test_x)

            r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]

            df = pd.DataFrame({
                'r_square': [r2],
                'axis': [['x', 'y']],
                'movement': 'velocity',
                'sourceSessionIndex': [sourceSessionIndex],
                'targetSessionIndex': [targetSessionIndex],
                'normalizedMethod': ['None'],
                'model': ['deepDomainComfuse'],
                'targetLength': targetLength
            })
            df = df.explode(['axis', 'r_square']).reset_index(drop=True)
            df.to_csv('./result_experiments/timeLengthTest.csv', index=False, header=False, mode='a')


            print(f'[domainAdversarial] targetLength = {targetLength}')
            ###########################################################################
            # DomainAdversarialNetwork
            model = DomainAdversarialNetwork()
            model.compile(optimizer='adam', run_eagerly=True)

            padding_width_m1 = [[0, sourceLength-targetLength], [0, 0], [0, 0]]
            train_x = {
                'source': sourceDataset['m1'][:sourceLength], 
                'target': np.pad(targetDataset['m1'][:targetLength], padding_width_m1, 'wrap') 
                }
            padding_width_movement = [[0, sourceLength-targetLength], [0, 0]]
            train_y = {
                'source_movement': sourceDataset['movement'][:sourceLength], 
                'target_movement': np.pad(targetDataset['movement'][:targetLength], padding_width_movement, 'wrap'), 
                'source_domain': tf.one_hot(np.zeros([sourceLength]), 2), 
                'target_domain': np.pad(tf.one_hot(np.ones([targetLength]), 2), padding_width_movement, 'wrap') 
                }

            test_x = {
                'source': np.zeros_like(targetDataset['m1'][5000:]), 
                'target': targetDataset['m1'][5000:]
                }
            test_y = targetDataset['movement'][5000:]

            train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
            train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128)  
            
            model.useTargetLabel = True
            epochs = 100
            for epoch in range(epochs):
                start_steps = epoch * len(train_dataset)
                total_steps = epochs * len(train_dataset)

                # logs = []

                for step, (x, y) in enumerate(train_dataset):
                
                    p = float(step + start_steps) / total_steps
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    model.alpha = alpha       
                    log = model.train_on_batch(x=x, y=y, return_dict=True)
                    # logs.append(log)
                
                # df = pd.DataFrame(logs)
                # print('epoch', epoch, 'decode_loss', df.mean()['decode_loss'], 'domain_loss', df.mean()['domain_loss'])

            # test
            model.predict_movement = True
            pred = model.predict(x=test_x)

            r2 = [r_square(pred=pred[:, i], true=test_y[:, i]) for i in range(pred.shape[-1])]

            df = pd.DataFrame({
                'r_square': [r2],
                'axis': [['x', 'y']],
                'movement': 'velocity',
                'sourceSessionIndex': [sourceSessionIndex],
                'targetSessionIndex': [targetSessionIndex],
                'normalizedMethod': ['None'],
                'model': ['domainAdversarial'],
                'targetLength': targetLength
            })
            df = df.explode(['axis', 'r_square']).reset_index(drop=True)
            df.to_csv('./result_experiments/timeLengthTest.csv', index=False, header=False, mode='a')

        