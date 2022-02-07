#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:19:11 2020

@author: ogk
"""


import torch.nn as nn
import torch.nn.functional as F
import param
from utils import ReverseLayerF


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.GRU(input_size=96, hidden_size=256, num_layers=1, 
                                batch_first=True, dropout=0.2, bidirectional=True)
        # nn.init.orthogonal_(self.extractor.weight_ih_l0, gain=1)
        # nn.init.orthogonal_(self.extractor.weight_hh_l0, gain=1)
        # nn.init.constant_(self.extractor.bias_ih_l0, 0)
        # nn.init.constant_(self.extractor.bias_hh_l0, 0)
        # self.relu_layer = nn.PReLU()
        # self.dim_reduction = nn.Linear(256, 64)
        self.flatten = nn.Flatten()
        self.reduction = nn.Linear(2560, 512)

    def forward(self, input_data):
        embedding, (h_s) = self.extractor(input_data)
        embedding = self.flatten(embedding)
        # embedding = self.relu_layer(embedding)  # [batch, tap_size, hiddem_dim*2]
        # embedding = self.dim_reduction(embedding) # [batch, tap_size, 64]
        # embedding = embedding.view(-1, 320) # [batch, tap_size*64]
        # # print('embedding', embedding.size())
        embedding = self.reduction(embedding)
        return embedding


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_x = nn.Sequential(
            # nn.Linear(2560, 512),
            # nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        )
        self.decoder_y = nn.Sequential(
            # nn.Linear(2560, 512),
            # nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embedding):
        output_x = self.decoder_x(embedding)
        output_y = self.decoder_y(embedding)
        return output_x, output_y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # nn.Linear(2560, 512),
            # nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU())
        self.fout = nn.Linear(64, 2)

    def forward(self, embedding, alpha):
        reversed_input = ReverseLayerF.apply(embedding, alpha)
        x = self.discriminator(reversed_input)
        x = self.fout(x)
        return x
    

class pre_discriminator(nn.Module):
    def __init__(self):
        super(pre_discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(640, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU())
        self.out = nn.Linear(64, 37)

    def forward(self, embedding):
        x = self.discriminator(embedding)
        x = self.out(x)
        return x