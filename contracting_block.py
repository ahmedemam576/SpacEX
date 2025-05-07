#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the Contracting Block for U-Net like architectures in GANs.

This module implements a standard contracting (downsampling) block, 
often used in the encoder part of a U-Net or similar generator/
discriminator architectures. It typically consists of a convolutional layer
that reduces spatial dimensions, followed by instance normalization (optional)
and an activation function.

Created on Wed Aug  3 11:27:45 2022
@author: ahmedemam576
"""

import torch
from torch import nn

class Contracting_Block(nn.Module):
    """
    A contracting (downsampling) block for neural network architectures.

    This block applies a 2D convolution to reduce the spatial dimensions
    (height and width) of the input feature map by a factor of 2 (due to stride=2)
    and doubles the number of channels. It can optionally apply instance
    normalization and uses either ReLU or LeakyReLU as the activation function.

    Attributes:
        conv (nn.Conv2d): The convolutional layer.
        use_inorm (bool): Flag indicating whether to use instance normalization.
        activation (nn.Module): The activation function (ReLU or LeakyReLU).
        instance_norm (nn.InstanceNorm2d, optional): The instance normalization layer,
                                                    present if use_inorm is True.
    """
    def __init__(self, input_channels, kernel_size=3, use_inorm=True, activation=\'relu\'):
        """
        Initializes the Contracting_Block.

        Args:
            input_channels (int): The number of channels in the input feature map.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            use_inorm (bool, optional): Whether to use instance normalization. Defaults to True.
            activation (str, optional): The type of activation function to use, 
                                      either \'relu\' or \'leaky_relu\'. Defaults to \'relu\'.
        """
        super(Contracting_Block, self).__init__()
        
        # Convolutional layer: doubles channels, halves spatial dimensions
        # padding_mode=\'reflect\' is often used in GANs to reduce border artifacts.
        self.conv = nn.Conv2d(
            input_channels,
            input_channels * 2,  # Output channels are double the input channels
            kernel_size=kernel_size,
            stride=2,            # Halves the spatial dimensions
            padding=1,           # (kernel_size - 1) // 2 for stride 2 to maintain size before striding if kernel is odd
            padding_mode=\'reflect\'
        )
        
        self.use_inorm = use_inorm
        
        # Activation function
        if activation == \'relu\':
            self.activation = nn.ReLU()
        elif activation == \'leaky_relu\': # Common to use LeakyReLU in GAN discriminators
            self.activation = nn.LeakyReLU(0.2) # 0.2 is a common negative slope for LeakyReLU
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose \'relu\' or \'leaky_relu\'.")

        # Instance Normalization layer (optional)
        # InstanceNorm2d is applied per channel per sample, common in style transfer and GANs.
        if use_inorm:
            self.instance_norm = nn.InstanceNorm2d(input_channels * 2)
            
    def forward(self, x):
        """
        Defines the forward pass of the Contracting_Block.

        Args:
            x (torch.Tensor): The input feature map.

        Returns:
            torch.Tensor: The output feature map after convolution, optional normalization, 
                          and activation.
        """
        x = self.conv(x)                # Apply convolution
        if self.use_inorm:
            x = self.instance_norm(x)   # Apply instance normalization if enabled
        x = self.activation(x)          # Apply activation function
        return x

