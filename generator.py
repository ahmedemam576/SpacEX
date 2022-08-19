 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:09 2022

@author: ahmedemam576
"""
import torch
from torch import nn

from expanding_block import Expanding_Block
#the expanding block divides the number of the filters by two, output_filters = input_filters//2
from contracting_block import Contracting_Block
#the contracting block doubles the number of the filters, output_filters = 2 * input_filters

from residual_block import Residual_Block
# the residual block consists of 2 convolution layers, and at the end we concatenate the input to the output
# also the input filters and the output filters are equal 
from feature_map import Feature_map
# the feature map class doesn't change the number of the filters , it's just a convolutional layer

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        
        super(Generator, self).__init__()
        self.up_feature =Feature_map(input_channels, hidden_channels)
        
        
        
        self.contract1 = Contracting_Block(hidden_channels)
        
        self.contract2 = Contracting_Block(hidden_channels*2)
        
        self.res1 = Residual_Block(hidden_channels*4)
        self.res2 = Residual_Block(hidden_channels*4)
        self.res3 = Residual_Block(hidden_channels*4)
        self.res4 = Residual_Block(hidden_channels*4)
        self.res5 = Residual_Block(hidden_channels*4)
        self.res6 = Residual_Block(hidden_channels*4)
        self.res7 = Residual_Block(hidden_channels*4)
        self.res8 = Residual_Block(hidden_channels*4)
        self.res9 = Residual_Block(hidden_channels*4)
        
        self.expand1 = Expanding_Block(hidden_channels*4)
        self.expand2 = Expanding_Block(hidden_channels*2)
        
        self.down_feature = Feature_map(hidden_channels, output_channels)
        self.tanh = nn.Tanh()
        self.init_weights()
        
        
    # initialize the parmaters weights    
    def init_weights(self):
        
        for m in self.modules():
       
            #print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                print('yes conv2d')
                torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.0)
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                print('yes batch norm')
                # these is a problem here
                print(m)
                torch.nn.init.normal_(m.weight, 0.0, 0.0)
                torch.nn.init.constant_(m.bias, 0)
                
        
    def forward(self, x):
        x= self.up_feature(x)
        x= self.contract1(x)
        x= self.contract2(x)
        x= self.res1(x)
        x= self.res2(x)
        x= self.res3(x)
        x= self.res4(x)
        x= self.res5(x)
        x= self.res6(x)
        x= self.res7(x)
        x= self.res8(x)
        x= self.res9(x)
        x= self.expand1(x)
        x= self.expand2(x)
        x= self.down_feature(x)
        x= self.tanh(x)
        return x
    
    
    
