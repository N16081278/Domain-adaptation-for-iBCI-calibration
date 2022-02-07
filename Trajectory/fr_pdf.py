#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:21:18 2021

@author: ogk
"""

import get_data
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

path = '/home/ogk/Documents/OGK/Trajectory/fr_PDF/'

# for session in range(1, 38):
#     bins = 50
#     fr = get_data.data_preprocess(session)
#     fr = np.sum(fr, axis=0)
#     fr.resize(len(fr), 1)
#     min_max_scaler = preprocessing.MinMaxScaler()
#     fr_norm = min_max_scaler.fit_transform(fr)
    
#     time = 1 # mins
#     duration_1 = int(time*(60*1000)/64)
#     time = 3 # mins
#     duration_3 = int(time*(60*1000)/64)
#     time = 5 # mins
#     duration_5 = int(time*(60*1000)/64)
    
    
#     sns.distplot(fr_norm[:duration_1, 0], color='b', bins = bins)
#     sns.distplot(fr_norm[:duration_3, 0], color='orange', bins = bins)
#     sns.distplot(fr_norm[:duration_5, 0], color='lime', bins = bins)
#     sns.distplot(fr_norm[duration_5:, 0], color='r', bins = bins)
#     plt.legend(['1 minute', '3 minutes', '5 minutes', 'else'])
#     plt.title('Session_'+str(session))
#     plt.ylabel('Probability density function')
#     plt.savefig(path+str(session)+'.png')
        
#     plt.cla()
#     plt.clf()
#     plt.close()

# pal = sns.blend_palette([sns.desaturate("royalblue", 0), "royalblue"], 37)
sessions = [i+1 for i in range(37)] 

plt.figure(figsize=(12, 10))
for session in range(37):
    bins = 50
    fr = get_data.data_preprocess(session+1)
    fr = np.sum(fr, axis=0)
    fr.resize(len(fr), 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    fr_norm = min_max_scaler.fit_transform(fr)
    
    sns.kdeplot(fr_norm[:, 0], legend=True)
    
    plt.legend(str(session+1))
plt.ylabel('Probability density function')
# plt.savefig(path+str(session)+'.png')
    
# plt.cla()
# plt.clf()
# plt.close()