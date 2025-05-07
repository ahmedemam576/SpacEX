#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier Loading and Activation Hooking Utility.

This script loads a pre-trained ResNet50 model from PyTorch Hub
and registers a forward hook to the final fully connected layer (fc).
This can be used to extract activations from that layer during a forward pass,
which is a common technique in model interpretability and feature analysis,
including methods like activation maximization.

Created on Mon Aug 22 13:36:30 2022
@author: ahmedemam576
"""

import torch
import torch.nn as nn

# --- Pre-trained Classifier Loading ---
# Load a ResNet50 model pre-trained on the ImageNet dataset (IMAGENET1K_V2 weights).
# ResNet50 is a widely used convolutional neural network architecture.
# `torch.hub.load` is a convenient way to load pre-trained models.
model = torch.hub.load(repository="pytorch/vision", model="resnet50", weights="IMAGENET1K_V2")

# --- Activation Hook Definition ---
# This section defines a mechanism to capture the output of a specific layer (activations)
# during the model's forward pass.

# `activations_dict` will store the activations. It should be initialized as an empty dictionary.
activations_dict = {}

def layer_hook(activation_storage_dict, layer_name):
    """
    Creates a hook function that saves the output of a layer.

    Args:
        activation_storage_dict (dict): A dictionary where the activations will be stored.
                                        The key will be `layer_name`.
        layer_name (str): The name to use as a key for storing the activations.

    Returns:
        function: The hook function to be registered.
    """
    def hook(module, input_tensor, output_tensor):
        """
        The actual hook function. This is called when the hooked layer is executed.
        It stores the layer's output tensor in the `activation_storage_dict`.

        Args:
            module (torch.nn.Module): The layer module itself.
            input_tensor (torch.Tensor or tuple of torch.Tensor): The input to the layer.
            output_tensor (torch.Tensor): The output from the layer.
        """
        activation_storage_dict[layer_name] = output_tensor
    return hook

# Register the forward hook to the fully connected layer ('fc') of the ResNet50 model.
# The `layer_hook` function is called with `activations_dict` (to store the output)
# and 'fc' (as the name for this activation).
# After a forward pass (e.g., `model(some_input_image)`), `activations_dict['fc']`
# will contain the output tensor of the 'fc' layer.
if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
    model.fc.register_forward_hook(layer_hook(activations_dict, 'fc'))
else:
    print("Warning: The model does not have a standard 'fc' layer (Linear) or it's named differently."
          "The hook for 'fc' layer activations was not registered.")

# Example of how to use it (optional, for testing):
# if __name__ == '__main__':
#     # Ensure the model is in evaluation mode if not training
#     model.eval()
#     # Create a dummy input tensor (batch_size, channels, height, width)
#     # ResNet50 expects 3-channel images, typically 224x224 for ImageNet pre-training.
#     dummy_input = torch.randn(1, 3, 224, 224)
#     # Perform a forward pass
#     with torch.no_grad(): # Disable gradient calculations for inference
#         output = model(dummy_input)
#     # Now, activations_dict should contain the 'fc' layer's output
#     if 'fc' in activations_dict:
#         print(f"Shape of 'fc' layer activations: {activations_dict['fc'].shape}")
#     else:
#         print("'fc' activations not found. Check hook registration.")
#     print(f"Model output shape: {output.shape}")


