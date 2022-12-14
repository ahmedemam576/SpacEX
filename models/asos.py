import random

import torch.nn as nn

from tlib import ttorch


class Model(ttorch.model.Module):
    """
    Neural network model: U-Net and classifier.

    :param in_channels: number of input channels, e.g. 10 for sentinel-2 images
    :param n_unet_maps: number of unet activation maps, e.g. 3; the following workflow can be run with 1, 2, or 3
        unet activation maps
    :param n_classes: number of output classes of the classifier; if 1, then only on number is outputted, then
        0 means anthropogenic and 1 means protected; if 2, then a vector with 2 entries is outputted, the first
        value describes the anthropogenic class and the second value the protected class
    :param unet_base_channels: number of unet base channels; the standard unet has 64
    :param double_conv: bool if double convolutions shall be performed in the unet; if false only one convolution in
        each step is performed; standard unet has True
    :param batch_norm: bool if batch norm shall be performed; standard unet has False
    :param unet_mode: mode for upsampling; can be 'bilinear', 'nearest' or None
    :param unet_activation: activation function for the last unet layer to get activation map; can be e.g. nn.Tanh()
    :param final_activation: activation function of the classifier; can be nn.Sigmoid() or nn.Softmax(dim=1)
    """

    def tinit(
            self,

            in_channels: int,
            n_unet_maps: int,
            n_classes: int,

            unet_base_channels: int,  # standard UNet has 64
            double_conv: bool,  # standard UNet has True
            batch_norm: bool,  # standard UNet has False
            unet_mode: str,  # 'bilinear', 'nearest' or None
            unet_activation,  # e.g. nn.Tanh()

            dropout: float,  # standard UNet has None
            final_activation,  # nn.Sigmoid() or nn.Softmax(dim=1)
    ):

        # unet
        
        unet_kwargs = {
            'in_channels': in_channels,
            'out_channels': n_unet_maps,
            'base_channels': unet_base_channels,
            'batch_norm': batch_norm,
            'double_conv': double_conv,
            'dropout': dropout,
            'final_activation': unet_activation,
        }

        if unet_mode is None:
            self.unet = ttorch.modules.unet.StandardUNet(**unet_kwargs)

        else:
            self.unet = ttorch.modules.unet.UpsamplingUNet(mode=unet_mode, **unet_kwargs)

        # random occlusion (just for training)
        self.random_occlusion = ttorch.modules.operations.RandomPixelOcclusion(probability=0.2)

        # classifier

        dropout = dropout if dropout is not None else 0

        self.classifier = nn.Sequential(
            #  nn.Dropout(p=dropout), do not run because of random occlusions
            nn.Conv2d(n_unet_maps, 2 * n_unet_maps, kernel_size=5, stride=3),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Conv2d(2 * n_unet_maps, 4 * n_unet_maps, kernel_size=5, stride=3),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Conv2d(4 * n_unet_maps, 8 * n_unet_maps, kernel_size=5, stride=3),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(512 * n_unet_maps, 128),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes),
        )

    def tforward(self, x):
        """
        Runs the whole network including unet and classifier without final activation.

        :param x: input tensor, e.g. Sentinel-2 image
        :return: non-activated classification prediction
        """

        # unet
        x = self.unet(x)

        # random occlusion with a chance of 50% if in training mode
        if self.training and random.choice([True, False]):
            x = self.random_occlusion(x)

        # classifier (without activation)
        x = self.classifier(x)

        return x

    def classify_unet_map(self, unet_map):
        """
        Runs only the classifier including final activation.

        :param unet_map: unet activation map
        :param pred: classification prediction
        """

        pred = self.classifier(unet_map)

        if self.hparams['final_activation'] is not None:
            pred = self.hparams['final_activation'](pred)

        return pred
