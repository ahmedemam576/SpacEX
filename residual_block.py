#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:30:48 2022

@author: ahmedemam576
one of the building blocks to the Gen and Disc

"""
import torch
from torch import nn

class Residual_Block(nn.Module):
    
    def __init__ (self, input_channels, kernel_size = 3 , use_inorm = True, activation= 'relu'):
        super(Residual_Block,self).__init__()
        self.conv1= nn.Conv2d(input_channels, input_channels, kernel_size,padding=1, padding_mode='reflect')
        self.conv2= nn.Conv2d(input_channels, input_channels, kernel_size,padding=1, padding_mode='reflect')
        if use_inorm:
            self.innorm = nn.InstanceNorm2d(input_channels)
        
        
    def forward(self, x):
        x_original = x.clone()
        x= self.conv1(x)
        x= self.innorm(x)
        
        if self.activation == 'relu':
            x= nn.functional.relu(x)
        elif self.activation == 'leakyrelu':
            x= nn.functional.leakyrelu(x,0.2)
            
        x= self.conv2(x)
        x= self.innorm(x)
        return x_original + x