#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:43:12 2020

@author: ogk
"""


import numpy as np

# source = 18
# target = 19

# num_session = 16
# select_label = 'vel'

channel = np.arange(96)+1
bin_width = 16
tap_size = 1
stopping = 50
split = True
repair_ch = True
pre_train_num = 15
training_len = 256
tuning_len = 2813
batch_size = 256
EPOCHS = 100

plot_len = 500