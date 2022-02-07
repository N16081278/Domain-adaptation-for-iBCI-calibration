#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 03:07:38 2021

@author: ogk
"""

import numpy as np
import torch
import new_get_data
import seaborn as sns
import matplotlib.pyplot as plt

a = np.load('mmd_cross_session(256).npy')
b = np.load('mmd_cross_session(1024).npy')
c = np.load('mmd_cross_session(2000).npy')

classes = ['0407_02', '0411_01', '0411_02', '0418_01', '0419_01', 
    '0420_01', '0426_01', '0622_01', '0624_03', '0627_01', 
    '0630_01', '0915_01', '0916_01', '0921_01', '0927_04', 
    '0927_06', '0930_02', '0930_05', '1005_06', '1006_02', 
    '1007_02', '1011_03', '1013_03', '1014_04', '1017_02', 
    '1024_03', '1025_04', '1026_03', '1027_03', '1206_02', 
    '1207_02', '1212_02', '1220_02', '0123_02', '0124_01', 
    '0127_03', '0131_02']
s = range(len(classes))
plt.figure(figsize=(16,12))
plt.plot(s, a[0, :], color='g', linewidth=5, label='256 samples')
plt.plot(s, b[0, :], color='b', linewidth=5, label='1024 samples')
plt.plot(s, c[0, :], color='r', linewidth=5, label='2048 samples')
plt.grid()
plt.legend(loc='lower right', fontsize=24)
plt.xticks(s, classes)
plt.xticks(rotation='vertical')
plt.xlabel('Session', fontsize=20)
plt.ylabel('Distance from 0407_02(session_1)', fontsize=20)
plt.title('Maximum Mean Discrepancy', fontsize=32)

# classes = ['0407_02', '0411_01', '0411_02', '0418_01', '0419_01', 
#     '0420_01', '0426_01', '0622_01', '0624_03', '0627_01', 
#     '0630_01', '0915_01', '0916_01', '0921_01', '0927_04', 
#     '0927_06', '0930_02', '0930_05', '1005_06', '1006_02', 
#     '1007_02', '1011_03', '1013_03', '1014_04', '1017_02', 
#     '1024_03', '1025_04', '1026_03', '1027_03', '1206_02', 
#     '1207_02', '1212_02', '1220_02', '0123_02', '0124_01', 
#     '0127_03', '0131_02']
# fig = plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
# ax = sns.heatmap(a, vmin=0, vmax=0.8, square=True, xticklabels=classes, yticklabels=classes, cmap='YlOrRd')
# ax.set_title('Maximum Mean Discrepancy', fontsize = 30)