import numpy as np
import pandas as pd
import os

def read_dataset(filePath:str, movementName:list, mode: str = 'channel'):
      
    data = pd.read_pickle(filePath)

    # setting filter        
    data = data[data['sensor'] == 'm1']                    
    data = data.sort_values(by=['timestamp'])
    data = data.reset_index(drop=True)
    
    spikeCount = np.array(data['spikeCount'].to_numpy().tolist())
   

    spikeCount = np.pad(spikeCount, [[29, 0], [0, 0], [0, 0]])
  
    spikeCount = [np.roll(np.expand_dims(spikeCount, 0), i, axis=1) for i in range(29, -1, -1)]
    spikeCount = np.concatenate(spikeCount, axis=0)
    spikeCount = spikeCount[:, 29:, :, :]
    spikeCount = np.transpose(spikeCount, (1, 0, 2, 3))

    if mode == 'channel':
        m1 = np.sum(spikeCount, axis=-1) 
    elif mode == 'unit':
        m1 = spikeCount.reshape((*spikeCount.shape[:-2], -1))    

   
    movement = [data[tuple(mv.split('_'))] for mv in movementName]
    movement = np.vstack(movement).T
    
    # type change
    m1 = m1.astype(np.float32)
    movement = movement.astype(np.float32)

    return {
        'm1': m1,
        'movement': movement,
        'timestamp': data['timestamp']
    }