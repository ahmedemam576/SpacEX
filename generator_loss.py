#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:35:22 2022

@author: ahmedemam576


"""
import torch
from torch.nn  import L1Loss, MSELoss
from gan_loss_term import Gan_loss_term

class Generator_Loss(Gan_loss_term):
    def __init__(self, real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y, adv_norm, identity_norm, cycle_norm, hook_dict):
        super(Generator_Loss,self).__init__(real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y, adv_norm, identity_norm, cycle_norm)
        
        self.hook_dict = hook_dict
        self.sum_adv_loss, _, _ , self.fake_X, self.fake_Y= self.adverserial_loss(self.real_X, self.real_Y, self.gen_XY, self.gen_YX, self.disc_X, self.disc_Y)
        #print('fake_y calculated')
        self.sum_identity_loss = self.identity_loss(self.real_X, self.real_Y, self.gen_XY, self.gen_YX)
        self.sum_cycle_loss = self.cycle_loss(self.real_X, self.real_Y, self.gen_XY, gen_YX)
   
        
        self.adverserial_loss(real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y)
        #print('adv inited-----------')
        self.identity_loss(real_X, real_Y, gen_XY, gen_YX)
        self.cycle_loss(real_X, real_Y, gen_XY, gen_YX)
        '''   self.sum_am_loss =self.AM_loss()'''
        
        
        
        
        
        
    
    def adverserial_loss(self, real_X, real_Y,gen_XY, gen_YX, disc_X, disc_Y):
        
        fake_Y =gen_XY(real_X)
        disced_fake_y = disc_Y(fake_Y)
        adv_loss_Y = (self.adv_norm(disced_fake_y, torch.ones_like(disced_fake_y)))
        
        fake_X =gen_YX(real_Y)
        disced_fake_X = disc_X(fake_X)
        adv_loss_X = (self.adv_norm(disced_fake_X, torch.ones_like(disced_fake_X)))
        
        sum_adv_loss = (adv_loss_X+adv_loss_Y)
        return (sum_adv_loss,adv_loss_Y, adv_loss_X, fake_X, fake_Y)
    
    
    
    
    def identity_loss(self, real_X,real_Y, gen_XY, gen_YX):
        id_x = gen_YX(real_X)
        id_loss_x = self.identity_norm(id_x, real_X)
        
        id_y = gen_XY(real_Y)
        id_loss_y = self.identity_norm(id_y, real_Y)
        
        sum_id_loss = id_loss_x+id_loss_y
        return sum_id_loss
    
    
    
   #' we use the adverserial_loss function to get the fake_x and fake_y used in the cycle loss'
    
    
    def cycle_loss(self, real_X,real_Y, gen_XY,gen_YX):
        
        fake_y= self.fake_Y
        cycle_XY = self.cycle_norm(real_Y, fake_y)
        
        fake_x = self.fake_X
        cycle_YX = self.cycle_norm(fake_x, real_X)
        
        sum_cycle_loss = cycle_XY+cycle_YX
        return sum_cycle_loss
    
    
    def AM_loss(self):
        #activation = self.hook_dict['fc'][0][340]
        activation = self.hook_dict[9][0][0]
        sum_am_loss =self.identity_norm(activation, torch.ones_like(activation))
        return sum_am_loss
    
    def __call__(self, adv_weight=1, id_weight=1, cycle_weight=1):
        
        x = (adv_weight * self.sum_adv_loss) + (id_weight * self.sum_identity_loss) + (cycle_weight * self.sum_cycle_loss) 
        #print('shape of sum_adv_loss ')
        return x
                
        
        
        
        
        
        
        
        
        