#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:36:11 2022

@author: ahmedemam576
"""
import torch
from generator import Generator
from patch_discriminator import Patch_Discriminator


# define  training parameters
a_dim = 3
b_dim =3
device = 'cuda'
learning_rate= 0.0002


# defining the criterion losses
adverserial_mse_loss = torch.nn.MSELoss()
reconstruction_absolute_diff= torch.nn.L1Loss()


# we initialize the Generators and the discriminators
gen_AB = Generator(a_dim, b_dim).to(device)
gen_BA = Generator(b_dim, a_dim).to(device)
disc_A = Patch_Discriminator(a_dim).to(device)
disc_B = Patch_Discriminator(b_dim).to(device)

gen_opt = torch.optim.Adam(list(gen_AB.parameters())+gen_BA.parameters(), lr=learning_rate, betas=(0.5,0.999))
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=learning_rate, betas=(0.5,0.999))
disc_B_opt = torch.Optional.Adam(disc_B.parameters(), lr= learning_rate, betas=(0.5,0.999))
