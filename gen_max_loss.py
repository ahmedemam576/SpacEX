#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:33:33 2022

@author: ahmedemam576
"""

'''maximizing generator loss function'''


import torch
from mutual_gen_loss import Mutual_Generator_Loss



class Max_Generator_Loss:
    def __init__(self, real_X, gen_max, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict, mined_x):
        #super().__init__(real_X, gen_max, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict)
        #self.neutralized_mined_x = neutralized_mined_x
        self.real_X= real_X
        #self.gen_max, self.gen_min, self.disc_min, self.disc_max =  gen_max, gen_min, disc_min, disc_max
        self.gen_max,  self.disc_min, self.disc_max =  gen_max, disc_min, disc_max
        self.hook_dict = hook_dict
        self.adv_norm = adv_norm
        self.identity_norm = identity_norm
        self.cycle_norm = cycle_norm
        self.mined_x = mined_x
        
       
        #self.adv_loss_max,self.maxed_x, self.mined_x= self.adverserial_loss(self.real_X, self.gen_max, self.gen_min, self.disc_min, self.disc_max)
        self.adv_loss_max,self.maxed_x=  self.adverserial_loss(self.real_X, self.gen_max, self.mined_x, self.disc_min, self.disc_max)
        
        #self.id_loss_max = self.identity_loss(self.maxed_x, self.mined_x, self.gen_max, self.gen_min)
        self.id_loss_max = self.identity_loss(self.maxed_x, self.mined_x, self.gen_max)
        
        #self.cycle_max = self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min, self.gen_max)
        self.cycle_max = self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_max)
        
        self.sum_actmax_loss =self.Act_max()
        
        
        
        self.Act_max()
        
        self.adverserial_loss(self.real_X, gen_max, mined_x, disc_min, disc_max)
        #print('adv inited-----------')
        #self.identity_loss(self.maxed_x, self.mined_x, self.gen_max, self.gen_min)
        self.identity_loss(self.maxed_x, self.mined_x, self.gen_max)
        
        #self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min, self.gen_max)
        self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_max)
        
        
    
        
        
    #def adverserial_loss(self, real_X, gen_max, gen_min, disc_min, disc_max):
    def adverserial_loss(self, real_X, gen_max, mined_x, disc_min, disc_max):    
        
        maxed_x =gen_max(real_X)
        disced_maxed_x = disc_max(maxed_x)
        adv_loss_max = (self.adv_norm(disced_maxed_x, torch.ones_like(disced_maxed_x)))
        
        #mined_x =gen_min(real_X)
        
        
        
    #    return (adv_loss_max, maxed_x, mined_x)
        return (adv_loss_max, maxed_x)
    
    
    #def identity_loss(self, maxed_x, mined_x, gen_max, gen_min):
    def identity_loss(self, maxed_x, mined_x, gen_max):
        
        
        id_maxed = gen_max(maxed_x)
        id_loss_max = self.identity_norm(id_maxed, maxed_x)
        
        
        return id_loss_max
    
    
    #def cycle_loss(self, real_X, maxed_x,mined_x, gen_min,gen_max):
    def cycle_loss(self, real_X, maxed_x,mined_x,gen_max):
        ''' 
        we try to force this function  gen_max(gen_min(x)) =====> x
        therefore by minimizing -> (gen_max(gen_min(x)) - x) , we create a cycle
        '''
        
        neutralized_mined_x = gen_max(mined_x)
        cycle_max = self.cycle_norm(neutralized_mined_x, real_X)
        
        return cycle_max
    
    
    def Act_max(self):
        #activation = self.hook_dict['fc'][0][340]
        activation = self.hook_dict[13][0][0]
        sum_actmax_loss =self.identity_norm(activation, torch.ones_like(activation))
        return sum_actmax_loss
    
    
    def __call__(self, adv_weight=0.35, id_weight=0.3, cycle_weight=0.2,activation_weight=0.18):
        
        x = (adv_weight * self.adv_loss_max) + (id_weight * self.id_loss_max) + (cycle_weight * self.cycle_max)+ (activation_weight*self.sum_actmax_loss)
        #x = (adv_weight * self.adv_loss_max) + (id_weight * self.id_loss_max) + (cycle_weight * self.cycle_max)
        #print('shape of sum_adv_loss ')
        return x