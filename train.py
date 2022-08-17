#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:36:11 2022

@author: ahmedemam576
"""
import torch
from generator import Generator
from patch_discriminator import Patch_Discriminator
import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        # glob searches for a file with specific pattern
        # join, just concatenate two pathes, and using ('sA' % mode) will add A at the end of the root path without spaces
        # sorted will give us the path sorted ascendingly
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the horse2zebra images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        'we are trying to solve the problem in case we have a greyscale image'
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        '''return (item_A - 0.5) * 2, (item_B - 0.5) * 2'''
        return item_A, item_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


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
