#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:53:03 2022

@author: ahmedemam576
"""

class Gan_loss_term:
    '''
    define the main building blocks used in many loss terms
    which is used commonly in GANs application
    ----> a constructor for other GANs loss functions
    '''
    def __init__ (self, real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y,  adv_norm, identity_norm, cycle_norm):
        
      
        self.gen_YX = gen_YX
        self.gen_XY = gen_XY
        self.disc_Y = disc_Y
        self. disc_X = disc_X
        self.real_X = real_X
        self.real_Y = real_Y
        self.adv_norm = adv_norm
        self. identity_norm= identity_norm
        self.cycle_norm = cycle_norm
        
        print(f'{self.name} is created')
        
        
            