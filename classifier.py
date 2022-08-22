#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:36:30 2022

@author: ahmedemam576
"""
''' in this file i would introduce the pretrained classifier '''
import torch

# registering a hook
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook
model.fc.register_forward_hook(layer_hook(dict, 'fc'))
