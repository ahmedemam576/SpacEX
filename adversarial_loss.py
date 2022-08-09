#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:43:26 2022

@author: ahmedemam576
"""
from gan_loss_term import Gan_loss_term
from torch import ones_like

class Adversarial_Loss(Gan_loss_term):
    def __init__(self,real_x, generator, discriminator, name, norm, weight=1.0):
        super(Adversarial_Loss,self).__init__(generator, discriminator, real_x, norm, name, weight)
        '''
        the class should inheret all his attributes from the Gan_loss_term
        so we shouldn't set each attribute to .self
        this loss should be used two time to produce fake_x and fake y using opposite generators'
        ''' 
        print(f'{self.name}is initiated')
    def __call__(self):
        fake_y =self.generator(self.real_x)
        disced_fake_y = self.discriminator(fake_y)
        adv_loss = self.weight*(self.norm(disced_fake_y, ones_like(disced_fake_y)))
      
        return adv_loss, fake_y
     
        