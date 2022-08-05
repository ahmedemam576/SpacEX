#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:30:03 2022

@author: ahmedemam576
one of the building blocks to the Gen and Disc

"""
import torch
from torch import nn

class Expanding_Block(nn.Module):
    def __init__ (self, input_channels, use_inorm= True, kernel_size =3, activation= 'relu'):
        
        super(Expanding_Block, self).__init__()
        self.tconv = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size, stride=2, padding= 1,output_padding=1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)#
        if use_inorm:
            self.inorm = nn.InstanceNorm2d(input_channels//2)
        
        def forward(self,x):
            x= self.tconv(x)
            if use_inorm:
                x= self.inorm(x)
            x=self.activation(x)
            return(x)
    
    
    
    
    