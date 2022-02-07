#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:43:12 2020

@author: ogk
"""


import numpy as np

# source = 18
# target = 19
num_session = 16
select_label = 'pos' # pos, vel, acc, fr

channel = np.arange(96)+1
bin_width = 16
tap_size = 5

training_len = 5600 ##
tuning_len = 2800
batch_size = 256
EPOCHS = 60

plot_len = 500