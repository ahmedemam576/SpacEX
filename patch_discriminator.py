#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:32:10 2022

@author: ahmedemam576
"""
import torch
from torch import nn
from contracting_block import Contracting_Block
from expanding_block import Expanding_Block
from residual_block import Residual_Block
from feature_map import Feature_map

class Patch_Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels =64):
        super(Patch_Discriminator,self).__init__()
        self.upfeature = Feature_map(input_channels, hidden_channels)
        self.contracting1 = Contracting_Block(hidden_channels,use_inorm=False,kernel_size =4, activation='leaky_relu')
        self.contracting2 = Contracting_Block(hidden_channels*2, kernel_size=4,activation='leaky_relu')
        self.contracting3= Contracting_Block(hidden_channels*4, kernel_size=4, activation='leaky_relu')
        self.finalConv = nn.Conv2d(hidden_channels*8, 1, kernel_size = 1)
        self.init_weights()
     
        
    # initialize the parmaters weights 
    def init_weights(self):
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.0)
            
        
    def Forward(self, x):
        x= self.upfeature(x)
        x=self.contracting1(x)
        x= self.contracting2(x)
        x= self.contracting3(x)
        x= self.finalConv(x)
        return x
    