#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:53:03 2022

@author: ahmedemam576
"""
from torch.nn  import L1Loss, MSELoss
class Gan_loss_term:
    '''
    define the main building blocks used in many loss terms
    which is used commonly in GANs application
    ----> a constructor for other GANs loss functions
    '''
    def __init__ (self, real_x, generator, discriminator, name, norm= 'l2',weight =1.0):
        
        self.weight = weight
        self.name  = name
        self. generator = generator
        self. discriminator = discriminator
        self.real_x = real_x
        print(f'{self.name} is created')
        if norm == 'l2':
            self.norm = MSELoss()
        elif norm =='l1':
            self.norm = L1Loss()
        else:
            self.norm =norm # for other norms
        
            