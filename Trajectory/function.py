#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:01:41 2020

@author: ogk
"""

import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def Get_Spike_Firring(firring_point, bins):
        spike_data = []
        time_last_point = bins[-1]
        for i in firring_point:
            if i < time_last_point:
                spike_data.append(i)
        # mapping_data = np.digitize(spike_data, bins)
        map_data, bin_arr = np.histogram(spike_data, bins=bins)
        
        return map_data.reshape(-1, len(map_data))
    
    
def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.
    
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = np.eye(num_classes)
    return y[labels-1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=(18,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt), fontsize=2, 
    #               horizontalalignment="center",
    #               color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
