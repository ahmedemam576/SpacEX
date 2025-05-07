#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the Adversarial Loss component for the CycleGAN model.

This module implements the adversarial loss, which is a crucial part of training
Generative Adversarial Networks (GANs). It encourages the generator to produce
outputs that are indistinguishable from real data by the discriminator.

Created on Tue Aug  9 10:43:26 2022
@author: ahmedemam576
"""

from .gan_loss_term import Gan_loss_term  # Import base class for GAN loss terms
from torch import ones_like                # PyTorch function to create a tensor of ones with the same shape as input

class Adversarial_Loss(Gan_loss_term):
    """
    Calculates the adversarial loss for a generator.

    This loss measures how well the generator can fool the discriminator.
    It is calculated based on the discriminator's output for the generated (fake) images.
    The goal is to make the discriminator classify fake images as real.

    Attributes:
        real_x (torch.Tensor): A batch of real images from the source domain.
        generator (torch.nn.Module): The generator network.
        discriminator (torch.nn.Module): The discriminator network.
        name (str): The name of this loss term (e.g., "AdversarialLoss_G_XtoY").
        norm (torch.nn.Module): The loss function to use (e.g., MSELoss, L1Loss).
        weight (float): The weight of this loss term in the total generator loss.
    """
    def __init__(self, real_x, generator, discriminator, name, norm, weight=1.0):
        """
        Initializes the Adversarial_Loss class.

        Args:
            real_x (torch.Tensor): A batch of real input images.
            generator (torch.nn.Module): The generator model.
            discriminator (torch.nn.Module): The discriminator model.
            name (str): Name for this loss component (e.g., "adversarial_loss_gen_AtoB").
            norm (torch.nn.Module): The loss function (e.g., nn.MSELoss()) to compare 
                                     discriminator output with target labels (all ones).
            weight (float, optional): Weighting factor for this loss. Defaults to 1.0.
        """
        super(Adversarial_Loss, self).__init__(generator, discriminator, real_x, norm, name, weight)
        # The class inherits all its attributes from the Gan_loss_term base class.
        # This loss is typically used twice in a CycleGAN setup: once for the generator
        # mapping from domain X to Y, and once for the generator mapping from Y to X.
        print(f'{self.name} is initiated')

    def __call__(self):
        """
        Computes the adversarial loss.

        The process involves:
        1. Generating fake images using the generator and real input images.
        2. Passing the fake images through the discriminator.
        3. Calculating the loss by comparing the discriminator's output for fake images
           to a tensor of ones (i.e., the target is for fake images to be classified as real).

        Returns:
            tuple: A tuple containing:
                - adv_loss (torch.Tensor): The calculated adversarial loss.
                - fake_y (torch.Tensor): The generated fake images.
        """
        # Generate fake images from the real input images
        fake_y = self.generator(self.real_x)
        
        # Get the discriminator's predictions for the fake images
        disced_fake_y = self.discriminator(fake_y)
        
        # Calculate the adversarial loss.
        # The target for the generator is to make the discriminator output ones (real) for fake images.
        # The loss is weighted by self.weight.
        adv_loss = self.weight * (self.norm(disced_fake_y, ones_like(disced_fake_y)))
      
        return adv_loss, fake_y
