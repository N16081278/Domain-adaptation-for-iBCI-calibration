import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score
from dataPreprocess import read_dataset
from model_tf import DeepDomainConfusionModel

folderPath = '../BCI_movementDecoder/data/indy_SUA'
fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('.pkl.zip')])


result = []
for sourceSessionIndex in range(36):
    targetSessionIndex = sourceSessionIndex + 1

    sourceSession = fileList[sourceSessionIndex]
    targetSession = fileList[targetSessionIndex]

    sourceDataset = read_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
    targetDataset = read_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    

    model = DeepDomainConfusionModel()
    model.compile(optimizer = 'adam', run_eagerly=True)

    TRAIN_COUNT = 5000

    source_m1 = sourceDataset['m1'][:, -5:, :]
    target_m1 = targetDataset['m1'][:, -5:, :]

    source_movement = sourceDataset['movement']
    target_movement = targetDataset['movement']

    test_timestamp = targetDataset['timestamp'][TRAIN_COUNT:]

    #
    train_x_source = source_m1[:TRAIN_COUNT]
    train_x_target = target_m1[:TRAIN_COUNT]
    train_y_source_movement = source_movement[:TRAIN_COUNT]
    train_y_target_movement = target_movement[:TRAIN_COUNT]
  

    test_x = {
        'source': np.zeros_like(target_m1[TRAIN_COUNT:]), 
        'target': target_m1[TRAIN_COUNT:]
        }

    test_y = target_movement[TRAIN_COUNT:]

    train_x = {
        'source': train_x_source, 
        'target': train_x_target
        }
    train_y = {
        'source_movement': train_y_source_movement, 
        'target_movement': train_y_target_movement
    }



    model.useTargetLabel = True
    #
    model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, shuffle=True)
  
    # 
    model.predict_movement = True
    target_train_pred = model.predict(x=train_x)
    source_train_pred = model.predict(x={'source': np.zeros([train_x_source.shape[0]]), 
                                           'target': train_x_source})

    target_test_pred = model.predict(x=test_x)
    source_test_pred = model.predict(x={'source': np.zeros([source_m1[TRAIN_COUNT:].shape[0]]), 
                                          'target': source_m1[TRAIN_COUNT:]})
    
    r2_source_train_axisX = r2_score(source_movement[:5000, 0], source_train_pred[:, 0])
    r2_source_train_axisY = r2_score(source_movement[:5000, 1], source_train_pred[:, 1])
    r2_source_test_axisX = r2_score(source_movement[5000:, 0], source_test_pred[:, 0])
    r2_source_test_axisY = r2_score(source_movement[5000:, 1], source_test_pred[:, 1])

    r2_target_train_axisX = r2_score(target_movement[:5000, 0], target_train_pred[:, 0])
    r2_target_train_axisY = r2_score(target_movement[:5000, 1], target_train_pred[:, 1])
    r2_target_test_axisX = r2_score(target_movement[5000:, 0], target_test_pred[:, 0])
    r2_target_test_axisY = r2_score(target_movement[5000:, 1], target_test_pred[:, 1])
    
    result.append([sourceSessionIndex, targetSessionIndex, r2_source_train_axisX, r2_source_train_axisY,
    r2_source_test_axisX, r2_source_test_axisY, r2_target_train_axisX, r2_target_train_axisY, 
    r2_target_test_axisX, r2_target_test_axisY])
    

col = ['sourceSessionIndex', 
       'targetSessionIndex', 
       ('train', 'source', 'x'),
       ('train', 'source', 'y'),
       ('test', 'source', 'x'),
       ('test', 'source', 'y'),
       ('train', 'target', 'x'),
       ('train', 'target', 'y'),
       ('test', 'target', 'x'),
       ('test', 'target', 'y')]
df = pd.DataFrame(result, columns=col)

df.to_csv('./ddc_v2.csv')

#

          