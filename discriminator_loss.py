#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:33:01 2022

@author: ahmedemam576
"""
import torch

class Discriminator_loss():
    
    '''
    real_x: oringinal image
    fake_x: a generated image from the generator..gen(AB)-->B`
    disc_x: a discriminatorfor the original image
    loss_type: L1 loss or MSELoss
    
    don't forget to add the weight when call the instance'
    
    '''
    def __init__(self, real_X, fake_X, disc_X, loss_type ):
        self.real_X = real_X
        self.fake_X = fake_X
        self.disc_X = disc_X
        self.loss_type = loss_type
        
        
    def __call__(self, weight):    
        disc_fake_x = disc_X(fake_X.detach()) # detach the generator, not to optimize the generator by the disc. objective function
        disc_fake_loss = loss_type(disc_fake_loss, torch.zeros_like(disc_fake_loss))
        disc_real_x = disc_X(real_X)
        disc_real_loss = loss_type(disc_real_loss, torch.ones_like(disc_real_loss))
        avg_disc_loss= 0.5*(disc_fake_loss + disc_real_loss)
        
        return weight* avg_disc_loss
    
    