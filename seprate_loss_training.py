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

# importing the framework's buildng blocks 
from generator import Generator
from patch_discriminator import Patch_Discriminator
from gen_min_loss import  Min_Generator_Loss
from gen_max_loss import Max_Generator_Loss     # ' separtate gen. losses'
from discriminator_loss import Discriminator_loss
from dataset import ZebraDataset


from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights: we use resnett 50 pretrained classifier trained on imagenet1k dataset
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to('cuda')

# registering a forward hook to the classifier to record the output of the last FC layer
def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook
hook_dict = dict()
model.fc.register_forward_hook(layer_hook(hook_dict, 'fc'))


''' change the dataloader to be able to produce only zebras'''
''' maximize the zebras activation neuron with index value =340 in the las FC layer'''

'''class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        # glob searches for a file with specific pattern
        # join, just concatenate two pathes, and using ('sA' % mode) will add A at the end of the root path without spaces
        # sorted will give us the path sorted ascendingly
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*')) # can be replaced by '/**', recursive = True
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
     
        return item_A, item_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))'''

adv_norm = nn.MSELoss() 
identity_norm = nn.L1Loss() 
cycle_norm =nn.L1Loss() 

n_epochs = 1000
dim_A = 3
dim_B = 3
display_step = 200
batch_size = 4
lr = 0.0002
load_shape = 286
target_shape = 256
device = 'cuda'

transform = transforms.Compose([
    transforms.Resize(load_shape),
    transforms.RandomCrop(target_shape),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    
                                ])




'''transform = transforms.Compose([
    
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''



path = 'horse2zebra'
mode= 'train'
dataset = ZebraDataset(path, mode, transform)


# define  training parameters
a_dim = 3
b_dim =3
device = 'cuda'
learning_rate= 0.0002
########################################################################################################









# defining the criterion losses

'''
we gonna sart the adaptaion for the generators and discriminator losses from here



'''



adverserial_mse_loss = torch.nn.MSELoss()
reconstruction_absolute_diff= torch.nn.L1Loss()


# initialize the Generators and the discriminators
gen_max = Generator(a_dim, b_dim).to(device)
gen_min = Generator(b_dim, a_dim).to(device)
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
            
            
        for real_A in dataloader:
            
            # image_width = image.shape[3]
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
            disc_max_opt.step() # Update optimizer
            
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



            # Keep track of the average discriminator loss
        
            mean_generator_loss =0
            mean_discriminator_loss_a += disc_max_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_max_loss.item() / display_step

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
                    
                show_tensor_images(torch.cat([real_A, real_A]), size=(dim_A, target_shape, target_shape))
                print('maxed_x shape =======>',maxed_x.shape)
                show_tensor_images(torch.cat([maxed_x, mined_x]), size=(dim_B, target_shape, target_shape))
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