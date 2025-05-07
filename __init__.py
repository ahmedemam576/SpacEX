#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main initialization file for the SpacEX package.

This file makes the core modules of the SpacEX CycleGAN implementation available
when the package is imported. It defines the `__all__` variable to specify
which modules are exported by `from SpacEX import *`.

@author: ahmedemam576
"""

# Import necessary modules for the CycleGAN model
from . import expanding_block        # Defines the expanding block used in the generator
from . import contracting_block      # Defines the contracting block used in the generator
from . import residual_block         # Defines the residual block used in the generator
from . import feature_map            # Defines the feature map layer
from . import generator              # Defines the generator model architecture
from . import generator_loss         # Defines the loss functions for the generator
from . import patch_discriminator    # Defines the PatchGAN discriminator model architecture
from . import discriminator_loss     # Defines the loss functions for the discriminator
from . import gan_loss_term          # Defines general GAN loss terms
from . import adversarial_loss, identity_loss, cycle_loss # Specific loss components

# Define the list of modules to be exported when `from SpacEX import *` is used.
__all__ = [
    'expanding_block', 'contracting_block', 'residual_block',
    'feature_map', 'generator', 'patch_discriminator', 'generator_loss', 'discriminator_loss',
    'gan_loss_term', 'adversarial_loss', 'identity_loss', 'cycle_loss'
]
