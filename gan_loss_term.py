#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:53:03 2022

@author: ahmedemam576
"""
import torch
class Gan_loss_term:
    '''
    define the main building blocks used in many loss terms
    which is used commonly in GANs application
    '''
    def __init__ (self, generator_out, discriminator_out, name, weight =1, norm= 'l2',):
        
        self.weight = weight
        self.name  = name
        self. generator_out = generator_out
        self. discriminator_out = discriminator_out
        
        if norm == 'l2':
            self.norm = torch.nn.MSELoss()
        elif norm =='l1':
            self.norm = torch.nn.L1Loss()
        else:
            self.norm =norm # for other norms
        
            