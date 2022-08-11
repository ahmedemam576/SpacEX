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
    def __init__ (self, real_X,real_Y, generator_XY, generator_YX, discriminator_X, discriminator_Y):
        
      
        self. generator = generator_YX
        self. generator = generator_XY
        self. discriminator = discriminator_Y
        self. discriminator = discriminator_X
        self.real_x = real_X
        self.real_x = real_Y
        print(f'{self.name} is created')
        
        
            