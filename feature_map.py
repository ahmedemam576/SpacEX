#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:40:35 2022

@author: ahmedemam576
one of the building blocks to the Gen and Disc
"""
import torch
from torch import nn
class Feature_map(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size =7):
        super(Feature_map,self).__init__()
        self.conv = nn.Conv2d(input_channels,output_channels, kernel_size = kernel_size, padding=3, padding_mode ='reflect')
    def forward(self, x ):
        x= self.conv(x)
        return x
        