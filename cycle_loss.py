#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the Cycle Consistency Loss component for the CycleGAN model.

This module implements the cycle consistency loss, a key component of CycleGAN.
It ensures that if an image is translated from domain X to domain Y and then
translated back from Y to X, the resulting image should be close to the original image.
This helps in learning meaningful mappings without paired training data.

Created on Tue Aug  9 16:15:35 2022
@author: ahmedemam576
"""

from .gan_loss_term import Gan_loss_term  # Import base class for GAN loss terms

class Cycle_Loss(Gan_loss_term):
    """
    Calculates the cycle consistency loss.

    This loss encourages the generators to be inverses of each other. 
    For example, if G_XY translates X to Y, and G_YX translates Y to X, then
    G_YX(G_XY(x)) should be close to x (forward cycle) and 
    G_XY(G_YX(y)) should be close to y (backward cycle).

    This class specifically calculates one direction of the cycle loss, e.g., || G_YX(G_XY(x)) - x ||.
    It expects `fake_y` (which is G_XY(real_x)) as input to its `__call__` method.

    Attributes:
        real_x (torch.Tensor): A batch of real images from the source domain (e.g., domain X).
        generator (torch.nn.Module): The generator that maps from the target domain back to the 
                                     source domain (e.g., G_YX if real_x is from domain X).
        name (str): The name of this loss term (e.g., "CycleLoss_XtoYtoX").
        norm (torch.nn.Module): The loss function to use for comparing the reconstructed image
                                with the original image (typically L1Loss).
        weight (float): The weight of this loss term (inherited from Gan_loss_term, but often 
                        a specific lambda_cycle is used in CycleGAN training).
    """
    def __init__(self, real_x, generator_YtoX, name, norm, weight=10.0):
        """
        Initializes the Cycle_Loss class.

        Args:
            real_x (torch.Tensor): A batch of real images from the original domain (e.g., X).
                                   This is what the reconstructed image will be compared against.
            generator_YtoX (torch.nn.Module): The generator that translates images from domain Y 
                                            back to domain X (e.g., G_YX).
            name (str): Name for this loss component (e.g., "cycle_loss_XtoYtoX").
            norm (torch.nn.Module): The loss function (e.g., nn.L1Loss()) to compare the 
                                     reconstructed image with the original real_x.
            weight (float, optional): Weighting factor for this cycle loss. 
                                      In CycleGAN, this is often denoted as lambda_cycle. 
                                      Defaults to 10.0, a common value.
        """
        # The `discriminator` argument in the base class is not strictly needed for cycle loss,
        # so we can pass None or the generator itself if the base class requires it.
        # For clarity, we are passing the generator_YtoX as the `generator` argument to the base class.
        super(Cycle_Loss, self).__init__(generator=generator_YtoX, discriminator=None, real_x=real_x, norm=norm, name=name, weight=weight)
        # Note: The base class `Gan_loss_term` might not be perfectly suited if it assumes a discriminator.
        # However, we are primarily using its structure for `name`, `norm`, `weight`, and `real_x`.
        # The `self.generator` in this class will refer to `generator_YtoX`.
        
        print(f'{self.name} is initiated with weight {self.weight}\
              (expects fake_y from G_XtoY as input to __call__, uses G_YtoX for reconstruction)')

    def __call__(self, fake_y):
        """
        Computes the cycle consistency loss for one direction.

        The process involves:
        1. Taking `fake_y` (which is the output of the other generator, e.g., G_XY(real_x)).
        2. Passing `fake_y` through this instance's generator (e.g., G_YX) to get `reconstructed_x`.
        3. Calculating the loss by comparing `reconstructed_x` with the original `self.real_x`.

        Args:
            fake_y (torch.Tensor): The generated image from the intermediate domain 
                                   (e.g., output of G_XtoY when original input was from domain X).

        Returns:
            torch.Tensor: The calculated cycle consistency loss, scaled by `self.weight`.
        """
        # `self.generator` here is generator_YtoX (or G_YX)
        # It takes fake_y (which is G_XY(real_x)) and tries to reconstruct real_x.
        reconstructed_x = self.generator(fake_y)
        
        # Calculate the cycle loss by comparing the reconstructed_x with the original real_x.
        # The loss is weighted by self.weight (lambda_cycle).
        cycle_loss = self.weight * self.norm(reconstructed_x, self.real_x)
        
        return cycle_loss

