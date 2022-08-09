#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:15:35 2022

@author: ahmedemam576
"""

from gan_loss_term import Gan_loss_term
class Cycle_Loss(Gan_loss_term):
    def __init__(self, real_x, generator, name, norm):
        super(Cycle_Loss, self).__init__(real_x, generator, name, norm)
        
        '''
        the class should inheret all his attributes from the Gan_loss_term
        so we shouldn't set each attribute to .self
        this loss should be used two time for opposite generators
        
        
        args:
            generator is Gen y--> x
            norm is L1 norm
            
            
        ''' 
        print(f'{self.name}is initiated')
        
    
    
    def __call__(self, fake_y):
        
        ''' fake y is tcalculated when calculating the adverserial loss in the direction from x->y '''
        gen_y_to_x = self.generator
        fake_x = gen_y_to_x(fake_y)
        cycle_loss= self.norm(fake_x, self.real_x)
        return cycle_loss