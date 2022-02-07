import numpy as np
import pandas as pd
import os
import h5py
import itertools
from scipy.interpolate import CubicSpline

mat_folder_path = './dataset/indy/Sorted_Spike_Dataset'

mat_files = [f for f in os.listdir(mat_folder_path) if f.endswith('.mat')]
mat_files.sort()

for mat_filename in mat_files:
    print(mat_filename)

    df = pd.DataFrame(columns=[])

    # Read Mat file
    mat = h5py.File(os.path.join(mat_folder_path, mat_filename), 'r')

    # raw data in .mat file
    spikes = mat['spikes']
    timestamp = mat['t'][()].flatten()
    original_interval = timestamp[1] - timestamp[0] # sec
    target_interval = 0.064 # sec


    # new (target) timestamp
    target_timestamp = np.arange(timestamp[0], timestamp[-1], target_interval)

    unit_cnt, channel_cnt = spikes.shape
    timestamp_cnt, = target_timestamp.shape

    # firing rate
    spikeCounts = np.empty([unit_cnt, channel_cnt, timestamp_cnt], dtype=np.int8)

    for unit, channel in itertools.product(range(unit_cnt), range(channel_cnt)):
        # get spike
        spike_data = mat[spikes[unit][channel]][()]
        if type(spike_data[0]) != np.ndarray:
            spike_data = np.array([])

        # firing rate calculate
        classification_arr = np.digitize(spike_data, target_timestamp)
        values, counts = np.unique(classification_arr, return_counts=True)
        
        spikeCount = np.zeros([timestamp_cnt +1])
        spikeCount[values] = counts

        spikeCounts[unit, channel, :-1] = spikeCount[1:-1]

    
    spikeCounts = spikeCounts.T

    spikeCounts_M1 = spikeCounts[:, :96, :]

    # M1
    arrays = [['spikeCount'] * spikeCounts_M1.shape[1], [f'channel_{c}' for c in range(spikeCounts_M1.shape[1])]]
    index = pd.MultiIndex.from_arrays(arrays)
    df_spikeCount = pd.DataFrame(spikeCounts_M1.tolist(), columns=index)

    df_spikeCount['sensor'] = 'm1'
    df_spikeCount['timestamp'] = target_timestamp

    df = df.append(df_spikeCount)

    # S1
    # if firingRates.shape[1] > 96:
    #     df = df.append(pd.DataFrame({
    #         'timestamp': target_timestamp, 
    #         'firingRate': firingRates[:, 96:, :].tolist(),        
    #         'sensor': 's1'
    #     }))
    
    
    ######################################################################################################
    ######################################################################################################

    finger_pos = mat['finger_pos']
    #  (z,-x,-y) to (x,y,z)
    finger_pos = np.vstack([-finger_pos[1,:], -finger_pos[2,:], finger_pos[0,:]])

    cs_x = CubicSpline(timestamp, finger_pos[0, :])
    cs_y = CubicSpline(timestamp, finger_pos[1, :])
    cs_z = CubicSpline(timestamp, finger_pos[2, :])

    # position
    finger_cs_pos = [cs_x(target_timestamp), cs_y(target_timestamp), cs_z(target_timestamp)]
    finger_cs_pos = np.vstack(finger_cs_pos)

    # velocity
    finger_cs_vel = [cs_x(target_timestamp,1), cs_y(target_timestamp,1), cs_z(target_timestamp,1)]
    finger_cs_vel = np.vstack(finger_cs_vel)  

    # acc
    finger_cs_acc = [cs_x(target_timestamp,2), cs_y(target_timestamp,2), cs_z(target_timestamp,2)]
    finger_cs_acc = np.vstack(finger_cs_acc)  

    # 調換維度順序
    finger_cs_pos = np.swapaxes(finger_cs_pos, 0, 1)
    finger_cs_vel = np.swapaxes(finger_cs_vel, 0, 1)
    finger_cs_acc = np.swapaxes(finger_cs_acc, 0, 1)

    array = [['position'] * 3, ['x', 'y', 'z']]
    df_position = pd.DataFrame(finger_cs_pos, columns=pd.MultiIndex.from_arrays(array))

    array = [['velocity'] * 3, ['x', 'y', 'z']]
    df_velocity = pd.DataFrame(finger_cs_vel, columns=pd.MultiIndex.from_arrays(array))

    array = [['acceleration'] * 3, ['x', 'y', 'z']]
    df_acceleration = pd.DataFrame(finger_cs_acc, columns=pd.MultiIndex.from_arrays(array))

    df_movement = pd.concat((df_position, df_velocity, df_acceleration), axis=1)
    df_movement['timestamp'] = target_timestamp

    df = df.merge(df_movement, on='timestamp', how='right')

    ######################################################################################################
    ######################################################################################################
    # Movement intention 運動意圖

    target_pos = mat['target_pos'][()].T
    intention_pos = [finger_pos.T[:, i] - target_pos[:, i] for i in range(2)]
    intention_pos = np.vstack(intention_pos).T
    
    cs_intention_x = CubicSpline(timestamp, intention_pos[:, 0])
    cs_intention_y = CubicSpline(timestamp, intention_pos[:, 1])

    # position
    intention_cs_pos = [cs_intention_x(target_timestamp), cs_intention_y(target_timestamp)]
    intention_cs_pos = np.vstack(intention_cs_pos)

    # velocity
    intention_cs_vel = [cs_intention_x(target_timestamp, 1), cs_intention_y(target_timestamp, 1)]
    intention_cs_vel = np.vstack(intention_cs_vel)  

    # acc
    intention_cs_acc = [cs_intention_x(target_timestamp, 2), cs_intention_y(target_timestamp, 2)]
    intention_cs_acc = np.vstack(intention_cs_acc)  

    # 調換維度順序
    intention_cs_pos = np.swapaxes(intention_cs_pos, 0, 1)
    intention_cs_vel = np.swapaxes(intention_cs_vel, 0, 1)
    intention_cs_acc = np.swapaxes(intention_cs_acc, 0, 1)

    array = [['intention_position'] * 2, ['x', 'y']]
    df_intention_position = pd.DataFrame(intention_cs_pos, columns=pd.MultiIndex.from_arrays(array))

    array = [['intention_velocity'] * 2, ['x', 'y']]
    df_intention_velocity = pd.DataFrame(intention_cs_vel, columns=pd.MultiIndex.from_arrays(array))

    array = [['intention_acceleration'] * 2, ['x', 'y']]
    df_intention_acceleration = pd.DataFrame(intention_cs_acc, columns=pd.MultiIndex.from_arrays(array))

    df_intention = pd.concat((df_intention_position, df_intention_velocity, df_intention_acceleration), axis=1)
    df_intention['timestamp'] = target_timestamp

    df = df.merge(df_intention, on='timestamp', how='right')

    print(df)
    exit()
    ######################################################################################################
    ######################################################################################################
    # save
    df.to_pickle(f'./data/{os.path.splitext(mat_filename)[0]}_{int(target_interval*1000)}ms.pkl.zip', \
        compression='zip')