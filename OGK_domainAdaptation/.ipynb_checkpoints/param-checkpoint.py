#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:43:12 2020

@author: ogk
"""


import numpy as np

select_label = 'vel'
mode = 'fix' # 'fix', 'retrain', 'mix', 'tune', 'DAN', 'DDC', 'MMD_tune'
target_label = True # False
channel = np.arange(96)+1
bin_width = 16
tap_size = 5
stopping = 50
split = True
repair_ch = True
training_len = 5000
# testing_len = 1250
batch_size = 128
test_bs = 512
EPOCHS = 100

# plot_len = 500