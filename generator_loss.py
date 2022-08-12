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
    def __init__(self, real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y, adv_norm, identity_norm, cycle_norm):
        super(Generator_Loss,self).__init__()
    
    
    def adverserial_loss (self,real_X, real_Y,gen_XY, gen_YX, disc_X, disc_Y, adv_norm):
        
        fake_Y =gen_XY(real_X)
        disced_fake_y = disc_Y(fake_Y)
        adv_loss_Y = (adv_norm(disced_fake_y, torch.ones_like(disced_fake_y)))
        
        fake_X =gen_YX(real_Y)
        disced_fake_X = disc_X(fake_X)
        adv_loss_X = (adv_norm(disced_fake_X, torch.ones_like(disced_fake_X)))
        avg_adv_loss = (adv_loss_X+adv_loss_Y)/2
        
      
        return (avg_adv_loss,adv_loss_Y, adv_loss_X, fake_X, fake_Y)
        
    
    def __call__(adv_weight, id_weight, cycle_weight):
        
        adverserial_loss =
        identity_loss =
        cycle_loss=
        
        
        
        
        
        
        
        
        adv_weight* adverserial_loss + id_weight*identity_loss+ cycle_weight* cycle_loss