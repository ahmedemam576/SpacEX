#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:05:19 2022

@author: ahmedemam576
"""
from gan_loss_term import Gan_loss_term
class Identity_Loss(Gan_loss_term):
    def __init__ (self, real_x, generator, discriminator, name):
        super(Identity_Loss,self).__init__(real_x, generator, discriminator, name)
        
        '''
        the class should inheret all his attributes from the Gan_loss_term
        so we shouldn't set each attribute to .self
        this loss should be used two time for opposite generators'
        ''' 
        print(f'{self.name}is initiated')
        
    def __call__(self):
        gen_y_to_x = self.generator
        iden_x= gen_y_to_x(self.real_x)
        
        id_loss = self.norm(iden_x, self.real_x)
        return id_loss, iden_x