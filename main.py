import argparse
from model_tf import DeepDomainConfusionModel
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os

def get_dataset(filePath:str, movementName:list):
      
    data = pd.read_feather(filePath)

    # setting filter        
    data = data[data['sensor'] == 'm1']                    
    data = data.sort_values(by=['timestamp'])
    data = data.reset_index(drop=True)

    # tapsize      
    data['firingRate'] = data['firingRate'].map(lambda x: np.array(x.tolist())) 
    for shift_time in range(1, 30):
        data[f'firingRate_lag_{shift_time}'] = data['firingRate'].shift(shift_time)
    data = data.dropna(axis=0)

    #       
    select_cols = [f'firingRate_lag_{i}' for i in range(29, 0, -1)]
    select_cols = select_cols + ['firingRate']   

    m1 = np.array([np.array(data[col].to_list()) for col in select_cols])
    m1 = np.sum(m1, axis=-1)
    m1 = np.swapaxes(m1, 0, 1)

    
    movement = [data[mv].to_numpy() for mv in movementName]
    movement = np.vstack(movement).T
    
    # type change
    m1 = m1.astype(np.float32)
    movement = movement.astype(np.float32)

    return {
        'm1': m1,
        'movement': movement,
        'timestamp': data['timestamp']
    }

def main():
    sourceSessionIndex = 13
    targetSessionIndex = 14

    folderPath = '../BCI_movementDecoder/data/indy'
    fileList = sorted([i for i in os.listdir(folderPath) if i.endswith('feather')])

    sourceSession = fileList[sourceSessionIndex]
    targetSession = fileList[targetSessionIndex]

    sourceDataset = get_dataset(os.path.join(folderPath, sourceSession), ['velocity_x', 'velocity_y'])
    targetDataset = get_dataset(os.path.join(folderPath, targetSession), ['velocity_x', 'velocity_y'])

    source_m1 = np.expand_dims(sourceDataset['m1'], axis=-1)
    target_m1 = np.expand_dims(sourceDataset['m1'], axis=-1)

    source_movement = np.expand_dims(sourceDataset['movement'], axis=-1)
    target_movement = np.expand_dims(targetDataset['movement'], axis=-1)
    TRAIN_COUNT = 5000
    train_x = np.concatenate((source_m1[:TRAIN_COUNT], target_m1[:TRAIN_COUNT]), axis=-1)
    train_y = np.concatenate((source_movement[:TRAIN_COUNT], target_movement[:TRAIN_COUNT]), axis=-1)
    test_x = target_m1[TRAIN_COUNT:]
    test_y = target_movement[TRAIN_COUNT:]

    model = DeepDomainConfusionModel()
    model.compile(optimizer = 'adam', run_eagerly=True)

    model.fit(x=train_x, y=train_y, batch_size=128, epochs=100, verbose=2)

    pass



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', default='finetune')
    # parser.add_argument('--source', default='<source dataset path>')
    # parser.add_argument('--target', default='<target dataset path>')
    # parser.add_argument('--')
    # args = parser.parse_args()

    # if args.mode == 'trainOnSource':
    #     pass
    # elif args.mode == 'finetune':
    #     pass
    # elif args.mode == 'trainOnTarget':
    #     pass
    # elif args.mode == 'mix':
    #     pass


    main()