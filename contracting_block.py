#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:27:45 2022

@author: ahmedemam576
one of the building blocks to the Gen and Disc

"""
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader


class Contracting_Block(nn.Module):
    def __init__(self,input_channels, kernel_size =3 ,use_inorm = True, activation = 'relu'):
        super(Contracting_Block,self).__init__()
        self.conv = nn.Conv2d(input_channels, input_channels*2, kernel_size = kernel_size, stride= 2, padding=1, padding_mode= 'reflect')
        self.use_inorm = use_inorm
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_inorm:
            self.instance_norm = nn.InstanceNorm2d(input_channels*2)
            
    def forward(self, x):
        x = self.conv(x)
        if self.use_inorm:
            x =self.instance_norm(x)
        x = self.activation(x)
        return x
        
    