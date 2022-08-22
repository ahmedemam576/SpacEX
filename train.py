#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:36:11 2022

@author: ahmedemam576
"""
import torch
import torchvision
from torch import nn
import glob
import random
import os


import torch
from generator import Generator
from patch_discriminator import Patch_Discriminator
from generator_loss import Generator_Loss
from discriminator_loss import Discriminator_loss

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights:
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to('cuda')

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[0] = output
    return hook
activation_dictionary = dict()
model.fc.register_forward_hook(layer_hook(activation_dictionary, 'fc'))




class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        # glob searches for a file with specific pattern
        # join, just concatenate two pathes, and using ('sA' % mode) will add A at the end of the root path without spaces
        # sorted will give us the path sorted ascendingly
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the horse2zebra images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        'we are trying to solve the problem in case we have a greyscale image'
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        '''return (item_A - 0.5) * 2, (item_B - 0.5) * 2'''
        return item_A, item_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))



adv_criterion = nn.MSELoss() 
recon_criterion = nn.L1Loss() 

n_epochs = 3
dim_A = 3
dim_B = 3
display_step = 200
batch_size = 1
lr = 0.0002
load_shape = 286
target_shape = 256
device = 'cuda'

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


dataset = ImageDataset("horse2zebra", transform=transform)


# define  training parameters
a_dim = 3
b_dim =3
device = 'cuda'
learning_rate= 0.0002


# defining the criterion losses
adverserial_mse_loss = torch.nn.MSELoss()
reconstruction_absolute_diff= torch.nn.L1Loss()


# we initialize the Generators and the discriminators
gen_AB = Generator(a_dim, b_dim).to(device)



gen_BA = Generator(b_dim, a_dim).to(device)
disc_A = Patch_Discriminator(a_dim).to(device)
disc_B = Patch_Discriminator(b_dim).to(device)

gen_opt = torch.optim.Adam(list(gen_AB.parameters())+list(gen_BA.parameters()), lr=learning_rate, betas=(0.5,0.999))
disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=learning_rate, betas=(0.5,0.999))
disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr= learning_rate, betas=(0.5,0.999))





''' you need to adapt this code into your architechture'''


def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss_a = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in  tqdm(range(n_epochs)):
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
        for real_A, real_B in dataloader:
            
            # image_width = image.shape[3]
            real_A = nn.functional.interpolate(real_A, size=target_shape)
            real_B = nn.functional.interpolate(real_B, size=target_shape)
             
            
            '''nn.functional.interpolate : Down/up samples the input to either the given size or the given scale_factor

            The algorithm used for interpolation is determined by mode.
            
            Currently temporal, spatial and volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D in shape.
            
            The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
            
            The modes available for resizing are: nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact'''
            
            
            cur_batch_size = len(real_A)
            
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            with torch.no_grad():
                
                outputs =model(real_A)
                activation = activation_dictionary[0][0][0]  #the first neuron in the linear layer
                print(activation)

            ### Update discriminator A ###
            disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
            disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            
            disc_a_loss = Discriminator_loss(real_A, fake_A, disc_A, adverserial_mse_loss)
                
                
            disc_a_loss = disc_a_loss()
            
            disc_a_loss.backward(retain_graph=True) # Update gradients
            disc_A_opt.step() # Update optimizer
            
            
            
            
            

            ### Update discriminator B ###
            
            with torch.no_grad():
                fake_B = gen_AB(real_A)
                
            disc_b_loss = Discriminator_loss(real_B, fake_A, disc_B, adverserial_mse_loss)
            disc_b_loss = disc_b_loss()
            disc_b_loss.backward(retain_graph=True) # Update gradients
            disc_B_opt.step() # Update optimizer
            
            
            
            

            ### Update generator ###
            gen_opt.zero_grad()
            main_generator_loss = Generator_Loss(real_X=real_A, real_Y=real_B,gen_XY= gen_AB, gen_YX=gen_BA,disc_X= disc_A,
            disc_Y=disc_B,adv_norm= adverserial_mse_loss,identity_norm= reconstruction_absolute_diff,cycle_norm= reconstruction_absolute_diff, hook_dict= activation_dictionary)
            
            main_generator_loss =main_generator_loss()
            #print('main_generator_loss----------------->',main_generator_loss.type)
            main_generator_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer



            # Keep track of the average discriminator loss
        
            mean_generator_loss =0
            mean_discriminator_loss_a += disc_a_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += main_generator_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator _a_ loss: {mean_discriminator_loss_a}")
                
                
                def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
                    '''
                    Function for visualizing images: Given a tensor of images, number of images, and
                    size per image, plots and prints the images in an uniform grid.
                    '''
                    image_tensor = (image_tensor + 1) / 2
                    image_shifted = image_tensor
                    image_unflat = image_shifted.detach().cpu().view(-1, *size)
                    image_grid = make_grid(image_unflat[:num_images], nrow=5)
                    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
                    plt.show()
                    
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss_a = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"cycleGAN_{cur_step}.pth")
            cur_step += 1
            
if __name__ == '__main__':
    
    train()