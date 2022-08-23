#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:30:36 2022

@author: ahmedemam576
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import torch
from torch.nn  import L1Loss, MSELoss
from gan_loss_term import Gan_loss_term

class Generator_Loss(Gan_loss_term):
    def __init__(self, real_X, gen_max, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict):
        super(Generator_Loss,self).__init__(real_X, gen_max, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm)
        
        self.hook_dict = hook_dict
        self.sum_adv_loss, _, _ , self.fake_X, self.fake_Y= self.adverserial_loss(self.real_X, self.real_Y, self.gen_XY, self.gen_YX, self.disc_X, self.disc_Y)
        #print('fake_y calculated')
        self.sum_identity_loss = self.identity_loss(self.real_X, self.real_Y, self.gen_XY, self.gen_YX)
        self.sum_cycle_loss = self.cycle_loss(self.real_X, self.real_Y, self.gen_XY, gen_YX)
        self.sum_adv_loss = self.AM_loss()
        
        self.adverserial_loss(real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y)
        #print('adv inited-----------')
        self.identity_loss(real_X, real_Y, gen_XY, gen_YX)
        self.cycle_loss(real_X, real_Y, gen_XY, gen_YX)
        self.sum_am_loss =self.AM_loss()
        
        
        
        
        
        
    
    def adverserial_loss(self, real_X, gen_max, gen_min, disc_min, disc_max):
        
        maxed_x =gen_max(real_X)
        disced_maxed_x = disc_max(maxed_x)
        adv_loss_max = (self.adv_norm(disced_maxed_x, torch.ones_like(disced_maxed_x)))
        
        mined_x =gen_min(real_X)
        disced_mined_x = disc_min(mined_x)
        adv_loss_min = (self.adv_norm(disced_mined_x, torch.ones_like(disced_mined_x)))
        
        sum_adv_loss = (adv_loss_max+adv_loss_min)
        
      
        return (sum_adv_loss,adv_loss_max, adv_loss_min, maxed_x, mined_x)
    
    
    
    
    def identity_loss(self, maxed_x, mined_x, gen_max, gen_min):
        id_mined = gen_min(mined_x)
        id_loss_min = self.identity_norm(id_mined, mined_x)
        
        id_maxed = gen_max(maxed_x)
        id_loss_max = self.identity_norm(id_maxed, maxed_x)
        
        sum_id_loss = id_loss_max + id_loss_min
        return sum_id_loss
    
    
    
   #' we use the adverserial_loss function to get the fake_x and fake_y used in the cycle loss'
    
    
    def cycle_loss(self, real_X, maxed_x,mined_x, gen_min,gen_max):
        ''' 
        we try to force this function  gen_max(gen_min(x)) =====> x
        therefore by minimizing -> (gen_max(gen_min(x)) - x) , we create a cycle
        '''
        
        neutralized_maxed_x = gen_min(maxed_x)
        cycle_max = self.cycle_norm(real_X, neutralized_maxed_x)
        
        
        neutralized_mined_x = gen_max(mined_x)
        cycle_min = self.cycle_norm(neutralized_mined_x, real_X)
        
        sum_cycle_loss = cycle_max+cycle_min
        return sum_cycle_loss
    
    
    def AM_loss(self):
        activation = self.hook_dict[0][0][0]
        sum_am_loss =self.identity_norm(activation, torch.ones_like(activation))
        return sum_am_loss
    
    def __call__(self, adv_weight=1, id_weight=1, cycle_weight=1):
        
        x = (adv_weight * self.sum_adv_loss) + (id_weight * self.sum_identity_loss) + (cycle_weight * self.sum_cycle_loss) 
        #print('shape of sum_adv_loss ')
        return x
                
        
        
        
        
        
        
        
        
        