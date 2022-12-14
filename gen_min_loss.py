#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:19:51 2022

@author: ahmedemam576
"""

'''minimizing generator loss function'''

import torch
from mutual_gen_loss import Mutual_Generator_Loss



class Min_Generator_Loss:
   # def __init__(self, real_X, gen_max, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict):
    def __init__(self, real_X, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict, maxed_x):   
        #super(Min_Generator_Loss,self).__init__(self, real_X, adv_norm, identity_norm, cycle_norm)

        
        self.maxed_x = maxed_x
        self.real_X= real_X
        #self.gen_max, self.gen_min, self.disc_min, self.disc_max =  gen_max, gen_min, disc_min, disc_max
        self.gen_min, self.disc_min, self.disc_max =  gen_min, disc_min, disc_max
        self.hook_dict = hook_dict
        self.adv_norm = adv_norm
        self.identity_norm = identity_norm
        self.cycle_norm = cycle_norm
        
        
       
        
        
        self.adv_loss_min, self.mined_x= self.adverserial_loss(self.real_X, self.maxed_x, self.gen_min, self.disc_min, self.disc_max)
        
        #self.id_loss_min = self.identity_loss(self.maxed_x, self.mined_x, self.gen_max, self.gen_min)
        self.id_loss_min = self.identity_loss(self.maxed_x, self.mined_x, self.gen_min)
        
        #self.cycle_min = self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min, self.gen_max)
        self.cycle_min = self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min)
        
   
        self.sum_actmin_loss =self.Act_min()
        self.Act_min()
        
        #self.adverserial_loss(real_X, gen_max, gen_min, disc_min, disc_max)
        self.adverserial_loss(self.real_X,self.maxed_x, self.gen_min, self.disc_min, self.disc_max)
        
        #print('adv inited-----------')
        #self.identity_loss(self.maxed_x, self.mined_x, self.gen_max, self.gen_min)
        self.identity_loss(self.maxed_x, self.mined_x, self.gen_min)
        
        #self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min, self.gen_max)
        self.cycle_loss(self.real_X, self.maxed_x,self.mined_x, self.gen_min)
        
        
        
    #def adverserial_loss(self, real_X, gen_max, gen_min, disc_min, disc_max):
    def adverserial_loss(self, real_X, maxed_x, gen_min, disc_min, disc_max):   
      #maxed_x =gen_max(real_X)
      
      
      mined_x =gen_min(real_X)
      disced_mined_x = disc_min(mined_x)
      adv_loss_min = (self.adv_norm(disced_mined_x, torch.ones_like(disced_mined_x)))
      
      #return (adv_loss_min, maxed_x, mined_x)
      return (adv_loss_min, mined_x)
  
    def identity_loss(self, maxed_x, mined_x, gen_min):
    #def identity_loss(self, maxed_x, mined_x, gen_max, gen_min):    
        id_mined = gen_min(mined_x)
        id_loss_min = self.identity_norm(id_mined, mined_x)
        return id_loss_min
    
    
    def cycle_loss(self, real_X, maxed_x,mined_x, gen_min):
    # def cycle_loss(self, real_X, maxed_x,mined_x, gen_min,gen_max):
        ''' 
        we try to force this function  gen_max(gen_min(x)) =====> x
        therefore by minimizing -> (gen_max(gen_min(x)) - x) , we create a cycle
        '''
        
        neutralized_maxed_x = gen_min(maxed_x)
        cycle_min = self.cycle_norm(real_X, neutralized_maxed_x)
    
    
        return cycle_min
    
    
    def Act_min(self):
        #activation = self.hook_dict['fc'][0][340]
        activation = self.hook_dict[13][0][0]
        sum_actmin_loss =self.identity_norm(activation, torch.zeros_like(activation))
        return sum_actmin_loss
    
    
    
    def __call__(self, adv_weight=0.35, id_weight=0.3, cycle_weight=0.2,activation_weight=0.18):
        
        x = (adv_weight * self.adv_loss_min) + (id_weight * self.id_loss_min) + (cycle_weight * self.cycle_min)+(activation_weight * self.sum_actmin_loss)
        #x = (adv_weight * self.adv_loss_min) + (id_weight * self.id_loss_min) + (cycle_weight * self.cycle_min)
        #print('shape of sum_adv_loss ')
        return x
        
        
        
        