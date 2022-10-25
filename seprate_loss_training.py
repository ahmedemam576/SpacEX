#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:34:13 2022

@author: ahmedemam576
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training for separate loss functions
"""
import torch
import torchvision
from torch import nn
import glob
import random
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# importing the framework's buildng blocks 
from generator import Generator
from patch_discriminator import Patch_Discriminator
from gen_min_loss import  Min_Generator_Loss
from gen_max_loss import Max_Generator_Loss     # ' separtate gen. losses'
from discriminator_loss import Discriminator_loss 
from dataset import ZebraDataset


from torchvision.models import resnet50, ResNet50_Weights

import warnings
import models.asos
from tlib import tlearn, ttorch, tutils
from tqdm import tqdm as tqdm_dataloader
from unet import UNet


run_wandb = False

if run_wandb:
    print('wandb initialization')
    wandb.init(project="max_project", entity="remote_sens")


# configuration

# paths

# the anthroprotect dataset (asos) can be downloaded here: http://rs.ipb.uni-bonn.de/data/anthroprotect/

hostname_username = tutils.machine.get_machine_infos()
print(hostname_username)

if hostname_username == ('ahmedemam576-Precision-7560', 'ahmedemam576'):  # ahmeds local machine
    working_dir = os.path.expanduser('~/working_dir')
    anthroprotect_data_path = '/home/ahmedemam576/working_folder/data/anthroprotect'
    mapinwild_data_path = os.path.expanduser('~/working_folder/mapinwild')

elif hostname_username == ('ibg2701', '?'):  # ahmeds box
    working_dir = '?'
    anthroprotect_data_path = '?'
    mapinwild_data_path = '?'

elif hostname_username == ('timodell', 'timo'):  # timos local machine
    working_dir = os.path.expanduser('~/working_dir')
    anthroprotect_data_path = os.path.expanduser('~/data/anthroprotect')
    mapinwild_data_path = os.path.expanduser('~/data/mapinwild')

elif hostname_username == ('ibg2701', 'tstomberg'):  # timos box
    working_dir = '/data/home/tstomberg/working_dir'
    anthroprotect_data_path = '/data/home/tstomberg/data/anthroprotect'
    mapinwild_data_path = '/data/home/aemam/datasets/mapinwild'

else:
    warnings.warn('No settings given for this computer/user!')


# define model

experiment = 'mapinwild'  # 'horse2zebra', 'anthroprotect', 'mapinwild'

if experiment == 'horse2zebra':
    # Using pretrained weights: we use resnett 50 pretrained classifier trained on imagenet1k dataset
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50(weights="IMAGENET1K_V1")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to('cuda')
    channels = list(range(3))

elif experiment in ['anthroprotect', 'mapinwild']:
    channels = list(range(3))  # specify accoring to model: if rgb: list(range(3)), if all: list(range(10))
    model = ttorch.model.load_model('./models/asos_mapinwild_rgb-channels.pt', Class=models.asos.Model)
    model.cuda()

else:
    warnings.warn('Unvalid string for model!')


# registering a forward hook to the classifier to record the output of the last FC layer
def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook
hook_dict = dict()

if experiment == 'horse2zebra':
    model.fc.register_forward_hook(layer_hook(hook_dict, 'fc'))

elif experiment in ['anthroprotect', 'mapinwild']:
    model.classifier[13].register_forward_hook(layer_hook(hook_dict, 9))




adv_norm = nn.MSELoss() 
identity_norm = nn.L1Loss() 
cycle_norm =nn.L1Loss() 

n_epochs = 3000
dim_A = len(channels)
dim_B = len(channels)
display_step = 200
batch_size = 1
lr = 0.0001
load_shape = 286 if experiment == 'horse2zebra' else 256

target_shape = 256
device = 'cuda'
num_workers = 6


if run_wandb:
    wandb.config = {
    "learning_rate": lr,
    "epochs": n_epochs,
    "batch_size": batch_size
                    }

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    
                                ])





if experiment == 'horse2zebra':
    path = 'horse2zebra'
    mode= 'train'
    dataset = ZebraDataset(path, mode, transform)

elif experiment in ['anthroprotect', 'mapinwild']:

    if experiment == 'anthroprotect':
        csv_file = os.path.join(anthroprotect_data_path, 'infos.csv')
        data_folder_tiles = os.path.join(anthroprotect_data_path, 'tiles', 's2')
    elif experiment == 'mapinwild':
        csv_file = os.path.join(mapinwild_data_path, 'tile_infos/file_infos.csv')
        data_folder_tiles = os.path.join(mapinwild_data_path, 'tiles')

    file_infos_df = pd.read_csv(csv_file)

    file_infos_df = file_infos_df[file_infos_df['label'] == 1]  # only protected areas
    if experiment == 'mapinwild':
        file_infos_df = file_infos_df[file_infos_df['subset'] == True]
        #file_infos_df = file_infos_df[file_infos_df['season'] == 'summer']

    datamodule = ttorch.data.images.DataModule(
        file_infos_df=file_infos_df,
        folder=data_folder_tiles,

        channels=channels,
        x_normalization=(0, 10000),
        clip_range=(0, 1),
        rotate=False,
        cutmix=None,
        n_classes=1,

        use_rasterio=True,
        rgb_channels=[2, 1, 0],

        batch_size=batch_size,
        num_workers=num_workers,
    )

    dataset = datamodule.train_dataset

# define  training parameters
a_dim = len(channels)
b_dim = len(channels)
device = 'cuda'
learning_rate= 0.0002
########################################################################################################



# defining the criterion losses

adverserial_mse_loss = torch.nn.MSELoss()
reconstruction_absolute_diff= torch.nn.L1Loss()


# initialize the Generators and the discriminators
#gen_max = Generator(a_dim, b_dim).to(device)
#gen_min = Generator(b_dim, a_dim).to(device)   # encoder decoder architecture for the generator

gen_max = UNet(a_dim,b_dim).to(device)   # changing encode decoder into unet 
gen_min = UNet(b_dim, a_dim).to(device)


disc_max = Patch_Discriminator(a_dim).to(device)
disc_min = Patch_Discriminator(b_dim).to(device)


# setting the optimizers for the gens and discs

gen_max_opt = torch.optim.Adam(gen_max.parameters(), lr=learning_rate, betas=(0.5,0.999))
gen_min_opt = torch.optim.Adam(gen_min.parameters(), lr=learning_rate, betas=(0.5,0.999))

disc_max_opt = torch.optim.Adam(disc_max.parameters(), lr=learning_rate, betas=(0.5,0.999))
disc_min_opt = torch.optim.Adam(disc_min.parameters(), lr= learning_rate, betas=(0.5,0.999))





''' you need to adapt this code into your architechture'''


def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss_a = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in  tqdm(range(n_epochs)):
        
        # Dataloader returns the batches
        # for image, _ in tqdm(dataloader):
            
            
        for real_A in tqdm_dataloader(dataloader):

            if experiment in ['anthroprotect', 'mapinwild']:
                real_A = real_A['x']
            
            # image_width = image.shape[3]
            if experiment == 'horse2zebra':
                real_A = nn.functional.interpolate(real_A, size=target_shape)
            
             
            
            '''nn.functional.interpolate : Down/up samples the input to either the given size or the given scale_factor

            The algorithm used for interpolation is determined by mode.
            
            Currently temporal, spatial and volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D in shape.
            
            The input dimensions are interpreted in the form: mini-batch x channels x [optional depth] x [optional height] x width.
            
            The modes available for resizing are: nearest, linear (3D-only), bilinear, bicubic (4D-only), trilinear (5D-only), area, nearest-exact'''
            
            
            cur_batch_size = len(real_A)
            
            real_A = real_A.to(device)
            
            model.eval() #we freeze the pretrained classifier weights 
            with torch.no_grad():
                model(real_A)
                #activation = activation_dictionary[0][0][0]  #the first neuron in the linear layer
                #print(activation)

            ### Update discriminator A ###
            
            '''
            lazm tl3b fel discriminators
            wel discriminator optimizer
            wel discriminator loss
            '''
            
            disc_min_opt.zero_grad() # Zero out the gradient before backpropagation
            disc_max_opt.zero_grad() # Zero out the gradient before backpropagation
            with torch.no_grad():
                mined_x = gen_min(real_A)
            
            disc_min_loss = Discriminator_loss(real_A, mined_x, disc_min, adv_norm)         
            disc_min_loss = disc_min_loss()
            
            disc_min_loss.backward(retain_graph=True) # Update gradients
            '''retain_graph=True ===> Right now, a real use case is multi-task learning where you have multiple losses that maybe be at different layers. 
            Suppose that you have 2 losses: loss1 and loss2 and they reside in different layers. In order to backprop the gradient of loss1 and loss2 w.r.t to the learnable weight 
            of your network independently. You have to use retain_graph=True in backward() method in the first back-propagated loss.'''
            disc_min_opt.step() # Update optimizer
            
            ### Update discriminator B ###
            with torch.no_grad():
                maxed_x = gen_max(real_A)
                
            disc_max_loss = Discriminator_loss(real_A, maxed_x, disc_max, adv_norm)
            disc_max_loss = disc_max_loss() #' running the call method'
            disc_max_loss.backward(retain_graph=True) # Update gradients
            disc_max_opt.step() # Update optimizer
            
            
            
            

            ### Update generator ###
            ''' hat8yr l generator 
            wel generator optimizer
            wel generator loss'''
            
            gen_max_opt.zero_grad()
            gen_min_opt.zero_grad()
            # add with torch no grad here to separate each gen from the 
            # computational graph of the other
            with torch.no_grad():
                mined_x = gen_min(real_A)
            gen_max_loss = Max_Generator_Loss(real_A, gen_max, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict, mined_x)
            with torch.no_grad():
                maxed_x = gen_max(real_A)
                #print('maxed_x shape =======>',maxed_x.shape)
            gen_min_loss = Min_Generator_Loss(real_A, gen_min, disc_min, disc_max, adv_norm, identity_norm, cycle_norm, hook_dict,maxed_x)
            # running the call method 
            gen_max_loss = gen_max_loss()
            gen_min_loss = gen_min_loss()
            
            # call the backward method
            gen_max_loss.backward()
            gen_min_loss.backward()
            
            # optimizers step
            gen_max_opt.step()
            gen_min_opt.step() # Update optimizer

            if run_wandb:
                wandb.log({
                    'disc_max_loss': disc_max_loss.item(),
                    'disc_min_loss': disc_min_loss.item(),
                    'gen_max_loss': gen_max_loss.item(),
                    'gen_min_loss': gen_min_loss.item(),
                }, step=cur_step)


            ### Visualization code ###
            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator _a_ loss: {mean_discriminator_loss_a}")
                
                
                if experiment == 'horse2zebra':
                
                    def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
                        '''
                        Function for visualizing images: Given a tensor of images, number of images, and
                        size per image, plots and prints the images in an uniform grid.
                        '''
                        image_shifted = image_tensor
                        image_unflat = image_shifted.detach().cpu().view(-1, *size).squeeze().numpy()
                        #print(f'image size =================<{image_unflat.shape}')
                        #image_grid = make_grid(image_unflat[:num_images], nrow=5)
                        image_grid= image_unflat.transpose(1, 2, 0).squeeze()
                        im = Image.fromarray((image_grid*255).astype(np.uint8))
                        
                        #print(f'image size =================<{image_unflat.shape}')
                        #       image_grid.save('myimage.jpg')

                        #plt.imshow(image_grid.permute(1, 2, 0).squeeze())
                        return im
                    maxed_images = show_tensor_images(maxed_x, size=(dim_A, target_shape, target_shape))
                    maxed_images.save('maxed_img_step{cur_step}_epoch{epoch}.jpg')
                    
                    mined_images = show_tensor_images(mined_x, size=(dim_A, target_shape, target_shape))
                    mined_images.save('mined_img_step{cur_step}_epoch{epoch}.jpg')
                    
                    real_images = show_tensor_images(real_A, size=(dim_A, target_shape, target_shape))
                    real_images.save('real_img_step{cur_step}_epoch{epoch}.jpg')
                    # we are just saving 3 images
                    if run_wandb:
                        print('logging with wandb')
                        wandb.log({f"maxed{epoch}{cur_step}": wandb.Image('maxed_img_step{cur_step}_epoch{epoch}.jpg')})  
                        wandb.log({f"mined{epoch}{cur_step}": wandb.Image('mined_img_step{cur_step}_epoch{epoch}.jpg')})  
                        wandb.log({f"real{epoch}{cur_step}": wandb.Image('real_img_step{cur_step}_epoch{epoch}.jpg')})
                        
                        ##################################################
                
                elif experiment in ['anthroprotect', 'mapinwild']:

                    def show_tensor_images(tensor, desc=''):
                        rgb = dataset.get_rgb(tensor[0].cpu())
                        plt.imshow(rgb)
                        #plt.show()
                        plt.savefig(desc + str(cur_step) + '.png')

                    # create folder
                    path = os.path.join(working_dir, 'images')
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    
                    show_tensor_images(real_A, os.path.join(working_dir, 'images/real'))
                    show_tensor_images(maxed_x, os.path.join(working_dir, 'images/maxed'))
                    show_tensor_images(mined_x, os.path.join(working_dir, 'images/mined'))
                    
                    if run_wandb:
                        wandb.log({f"maxed{epoch}{cur_step}": wandb.Image(os.path.join(working_dir, f'images/maxed{cur_step}.png'))})  
                        wandb.log({f"mined{epoch}{cur_step}": wandb.Image(os.path.join(working_dir, f'images/mined{cur_step}.png'))})  
                        wandb.log({f"real{epoch}{cur_step}": wandb.Image(os.path.join(working_dir, f'images/real{cur_step}.png'))})
                
                mean_generator_loss = 0
                mean_discriminator_loss_a = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_max': gen_max.state_dict(),
                        'gen_min': gen_min.state_dict(),
                        'gen_max_opt': gen_max_opt.state_dict(),
                        'gen_min_opt': gen_min_opt.state_dict(),
                        'disc_min': disc_min.state_dict(),
                        'disc_min_opt': disc_min_opt.state_dict(),
                        'disc_max': disc_max.state_dict(),
                        'disc_max_opt': disc_max_opt.state_dict()
                    }, f"cycleGAN_{cur_step}.pth")
            cur_step += 1
            
if __name__ == '__main__':
    
    train()