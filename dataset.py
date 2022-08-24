#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:32:18 2022

@author: ahmedemam576
"""
from torch.utils.data import Dataset 
import glob
import os
from PIL import Image
import torch
from skimage.color import gray2rgba
import numpy as np

path = 'horse2zebra'
mode = 'train'


class ZebraDataset(Dataset):
    def __init__(self, path, mode, transform):
        
        super(ZebraDataset, self).__init__()
        self.horse_file =   sorted(glob.glob(os.path.join(path, '%sB' % mode) + '/*.*'))
        assert len(self.horse_file)> 0 
        self.transform = transform
    
    def __len__(self):
        return len(self.horse_file)
        
    def __getitem__(self, index):
        img = self.transform(Image.open(self.horse_file[index]))
        shape = np.array(img).shape
        
        return img
        
        