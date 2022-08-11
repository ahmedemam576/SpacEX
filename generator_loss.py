#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:35:22 2022

@author: ahmedemam576
"""
from torch.nn  import L1Loss, MSELoss
from gan_loss_term import Gan_loss_term

class Generator_Loss(Gan_loss_term)):
    def __init__(self, real_X, real_Y, gen_XY, gen_YX, disc_X, disc_Y, adv_norm, identity_norm, cycle_norm):
    super(Generator_Loss,self).__init__()
    def        
    
    