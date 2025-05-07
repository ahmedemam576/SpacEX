#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loading utility for CycleGAN.

This script defines a PyTorch Dataset class for loading images for a CycleGAN model.
It is designed to load images from a specified directory structure, typically used
in image-to-image translation tasks where domains A and B are in separate subfolders.

This particular implementation seems to be tailored for one direction (e.g., loading images from domain B - zebras)
and was originally named ZebraDataset. It can be adapted for a more general CycleGAN dataset
that loads pairs of unaligned images from two domains.

Created on Wed Aug 24 13:32:18 2022
@author: ahmedemam576
"""

from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch
import numpy as np

class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading images from a single domain for CycleGAN.

    This dataset loads all images from a specified subfolder (e.g., trainA, trainB, testA, testB)
    and applies a given transformation. It ensures images are converted to RGB format.

    Attributes:
        files (list): A sorted list of file paths to the images in the specified domain.
        transform (callable): A torchvision.transforms callable to be applied to each image.
    """
    def __init__(self, root_dir, domain_subfolder=\"trainB\", transform=None):
        """
        Initializes the ImageDataset.

        Args:
            root_dir (str): The root directory of the dataset (e.g., path to \'horse2zebra\').
            domain_subfolder (str, optional): The subfolder within `root_dir` containing the images
                                            for the desired domain and split (e.g., \"trainA\", \"trainB\",
                                            \"testA\", \"testB\"). Defaults to \"trainB\".
            transform (callable, optional): A function/transform to apply to the images.
                                          Typically a `torchvision.transforms.Compose` object.
                                          Defaults to None.
        """
        super(ImageDataset, self).__init__()
        # Construct the path to the image files for the specified domain and mode (e.g., path/to/dataset/trainB/*.*)
        self.files = sorted(glob.glob(os.path.join(root_dir, domain_subfolder) + \"/*.*\"))
        if not self.files:
            print(f"Warning: No files found in {os.path.join(root_dir, domain_subfolder)}")
        
        self.transform = transform
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.files)
        
    def __getitem__(self, index):
        """
        Retrieves an image by its index, applies transformations, and returns it.

        Args:
            index (int): The index of the image to retrieve.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        try:
            img = Image.open(self.files[index])
        except Exception as e:
            print(f"Error opening image {self.files[index]}: {e}")
            # Return a dummy tensor or handle error as appropriate
            # For simplicity, returning None here, but robust error handling is better.
            # If transform expects a PIL image, this will cause issues downstream.
            # A placeholder black image could be returned instead.
            if self.transform:
                return self.transform(Image.new("RGB", (256, 256), (0,0,0))) # Example placeholder
            return None 

        # Ensure the image is in RGB format (3 channels)
        if img.mode != \"RGB\": 
            img = img.convert(\"RGB\")
        
        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
        else:
            # If no transform, convert PIL image to tensor manually (example)
            # This is a basic conversion; usually, normalization and other steps are in self.transform
            img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            
        return img

# Example Usage (commented out):
# if __name__ == \'__main__\':
#     from torchvision import transforms
#     # Define example transformations
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     # Create a dummy dataset structure for testing
#     if not os.path.exists(\'./dummy_dataset/trainA\'):
#         os.makedirs(\'./dummy_dataset/trainA\')
#         Image.new(\'RGB\', (100,100)).save(\'./dummy_dataset/trainA/dummy1.jpg\')
#     if not os.path.exists(\'./dummy_dataset/trainB\'):
#         os.makedirs(\'./dummy_dataset/trainB\')
#         Image.new(\'RGB\', (100,100)).save(\'./dummy_dataset/trainB/dummy2.jpg\')

#     # Test dataset for domain A
#     dataset_A = ImageDataset(root_dir=\'./dummy_dataset\
#                              , domain_subfolder=\'trainA\', transform=transform)
#     if len(dataset_A) > 0:
#         sample_A = dataset_A[0]
#         print(f"Sample A shape: {sample_A.shape}")
#     else:
#         print("Dataset A is empty.")

#     # Test dataset for domain B
#     dataset_B = ImageDataset(root_dir=\'./dummy_dataset\
#                              , domain_subfolder=\'trainB\', transform=transform)
#     if len(dataset_B) > 0:
#         sample_B = dataset_B[0]
#         print(f"Sample B shape: {sample_B.shape}")
#     else:
#         print("Dataset B is empty.")

