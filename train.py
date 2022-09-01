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

import warnings
import asos_model
from tlib import tlearn, ttorch, tutils
from tqdm import tqdm as tqdm_dataloader


# configuration

experiment = 'asos'  # 'resnet', 'asos'
channels = list(range(10))  # list(range(10)) means take all channels, for RGB give list [0, 1, 2]


# paths

# the anthroprotect dataset (asos) can be downloaded here: http://rs.ipb.uni-bonn.de/data/anthroprotect/
# the model state dict of asos can be downloaded here: http://rs.ipb.uni-bonn.de/downloads/asos/

hostname_username = tutils.machine.get_machine_infos()

if hostname_username == ('?', '?'):  # ahmeds local machine
    asos_model_checkpoint = '?'
    asos_data_path = '?'

elif hostname_username == ('cubesat.itg.uni-bonn.de', '?'):  # ahmeds box
    asos_model_checkpoint = '?'
    asos_data_path = '?'

elif hostname_username == ('timodell', 'timo'):  # timos local machine
    asos_model_checkpoint = os.path.expanduser('~/working_dir/model_state_dict.pt')
    asos_data_path = os.path.expanduser('~/data/anthroprotect')

elif hostname_username == ('cubesat.itg.uni-bonn.de', 'tstom'):  # timos box
    asos_model_checkpoint = '/scratch/tstom/working_dir/model_state_dict.pt'
    asos_data_path = '/scratch/tstom/data/anthroprotect'

else:
    warnings.warn('No settings given for this computer/user!')


# define model

if experiment == 'resnet':
    # Using pretrained weights:
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50(weights="IMAGENET1K_V1")

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to('cuda')

elif experiment == 'asos':
    model = asos_model.Model(
        in_channels=len(channels), n_unet_maps=3, n_classes=1, unet_base_channels=32, double_conv=False, batch_norm=True,
        unet_mode='bilinear', unet_activation=nn.Tanh(), final_activation=nn.Sigmoid())
    model.load_state_dict(torch.load(asos_model_checkpoint))
    model.cuda()

else:
    warnings.warn('Unvalid string for model!')


# set hook

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[0] = output
    return hook
activation_dictionary = dict()

if experiment == 'resnet':
    model.fc.register_forward_hook(layer_hook(activation_dictionary, 'fc'))

elif experiment == 'asos':
    model.classifier[9].register_forward_hook(layer_hook(activation_dictionary, 9))


# dataset

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

n_epochs = 5
dim_A = 3 if experiment == 'resnet' else 10
dim_B = 3 if experiment == 'resnet' else 10
display_step = 200
batch_size = 1
lr = 0.0002
load_shape = 286 if experiment == 'resnet' else 256
target_shape = 256
device = 'cuda'
num_workers = 6

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


class ImageDatasetASOS(Dataset):

    def __init__(self):
        
        csv_file = os.path.join(asos_data_path, 'infos.csv')
        data_folder_tiles = os.path.join(asos_data_path, 'tiles', 's2')

        file_infos = tlearn.data.files.FileInfosGeotif(
            csv_file=csv_file,
            folder=data_folder_tiles,
        )

        datamodule = ttorch.data.images.DataModule(
            file_infos=file_infos.df,
            folder=data_folder_tiles,

            channels=channels,
            x_normalization=(0, 10000),
            clip_range=(0, 1),
            rotate=False,
            cutmix=None,
            n_classes=1,

            use_rasterio=True,
            rgb_channels=[2, 1, 0],
            val_range=(0, 2**10),

            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.dataset = datamodule.train_dataset

    def __getitem__(self, index):

        return self.dataset[index]['x'], self.dataset[index]['x']  # two times same image

    def __len__(self):
        return len(self.dataset)


if experiment == 'resnet':
    dataset = ImageDataset("horse2zebra", transform=transform)

elif experiment == 'asos':
    dataset = ImageDatasetASOS()


# define  training parameters
a_dim = 3 if experiment == 'resnet' else len(channels)
b_dim =3 if experiment == 'resnet' else len(channels)
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
            
        '''hat8yr fel dataloader b7es ytl3 sora wa7da bs'''
        for real_A, real_B in tqdm_dataloader(dataloader, desc='current epoch', leave=False):
            
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
                #activation = activation_dictionary[0][0][0]  #the first neuron in the linear layer
                #print(activation)

            ### Update discriminator A ###
            
            '''
            lazm tl3b fel discriminators
            wel discriminator optimizer
            wel discriminator loss'''
            
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
            ''' hat8yr l generator 
            wel generator optimizer
            wel generator loss'''
            
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
                
                if experiment == 'resnet':

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
                
                elif experiment == 'asos':

                    def show_tensor_images(tensor, desc=''):
                        rgb = dataset.dataset.get_rgb(tensor[0].cpu())
                        plt.imshow(rgb)
                        #plt.show()
                        plt.savefig(desc + str(cur_step) + '.png')

                    show_tensor_images(real_A, os.path.expanduser('~/working_dir/images/real_A'))
                    show_tensor_images(fake_A, os.path.expanduser('~/working_dir/images/fake_A'))
                
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
    
    train(True)