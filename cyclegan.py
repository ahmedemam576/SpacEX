
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


#####
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#from models_file import *
#from datasets import *
#from utils import *

import models.asos
from tlib import tlearn, ttorch, tutils
from tqdm import tqdm as tqdm_dataloader


import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="horse2zebra", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=5, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=5.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=13.0, help="identity loss weight")
parser.add_argument("--lambda_act_max", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

import warnings
#import asos_model
from tlib import tlearn, ttorch, tutils
from tqdm import tqdm as tqdm_dataloader

channels = list(range(3)) 

#asos_data_path = os.path.expanduser('~/datasets/mapinwild')
asos_data_path = os.path.expanduser('~/data/mapinwild')
csv_file = os.path.join(asos_data_path, 'tile_infos/file_infos.csv')
data_folder_tiles = os.path.join(asos_data_path, 'tiles')

def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook
hook_dict = dict()


experiment = 'mapinwild'
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








experiment = 'mapinwild'
if experiment == 'horse2zebra':
    model.fc.register_forward_hook(layer_hook(hook_dict, 'fc'))

elif experiment in ['anthroprotect', 'mapinwild']:
    model.classifier[9].register_forward_hook(layer_hook(hook_dict, 9))
    model.eval()


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
    val_range=(0, 2**10),

    batch_size=opt.batch_size,
    num_workers=opt.n_cpu,
)

dataset = datamodule.train_dataset
######










# Create sample and checkpoint directories
os.makedirs("images_mapinwild_act_max/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models_mapinwild_act_max/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
#optimizer_G = torch.optim.Adam(
#    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
#)

optimizer_G_AB = torch.optim.Adam(
    G_AB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_G_BA = torch.optim.Adam(
    G_BA.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)


optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
'''lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step    
)'''

lr_scheduler_G_AB = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G_AB, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

lr_scheduler_G_BA = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G_BA, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step 
    )






lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
'''transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.Resampling.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]'''

# Training data loader
dataloader = DataLoader(
    dataset,
    #ImageDataset('/home/ahmedemam576/A/*.*', transforms_=transforms_, unaligned=True) ,
    batch_size=3,
    shuffle=False,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    #ImageDataset('/home/ahmedemam576/A/*.*', transforms_=transforms_, unaligned=True),
    dataset,
    batch_size=5,
    shuffle=False,
    num_workers=1,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["x"].type(Tensor))
    #real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["x"].type(Tensor))

    #real_B = Variable(imgs["A"].type(Tensor))
    fake_A = G_BA(real_B)
    difference = fake_B - fake_A
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    #real_B = make_grid(real_B, nrow=5, normalize=True)
    difference = make_grid(difference, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=False)
    fake_B = make_grid(fake_B, nrow=5, normalize=False)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_A, fake_B, difference), 1)
    save_image(image_grid, "images_mapinwild_act_max/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------
def train_loop():
    
    prev_time = time.time()
    #print('start___________>',prev_time)
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            #print('start in batch loop___________>',prev_time)
            
            # Set model input
            #real_A = Variable(batch["A"].type(Tensor))
            #real_B = Variable(batch["A"].type(Tensor))
            real_A = Variable(batch["x"].type(Tensor))
            real_B = Variable(batch["x"].type(Tensor))
            
            
            model(real_A).retain_graph=True #forward path of the image to the input
            
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)
    
            # ------------------
            #  Train Generators
            # ------------------
    
            G_AB.train()
            G_BA.train()
    
            #optimizer_G.zero_grad()
            optimizer_G_AB.zero_grad()
            optimizer_G_BA.zero_grad()
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
    
            #loss_identity = (loss_id_A + loss_id_B) / 2
    
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)#.retain_graph=True
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
    
            #loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
    
            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
    
            #loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
            
            # activation_maximization_loss
            activation=   hook_dict[9][0][0]
            print('activation --------------------->', activation.data)
            loss_AM_AB = criterion_identity (activation, torch.ones_like(activation)) # maximzing the wilderness class minimize(output -1)
            loss_AM_BA = criterion_identity (activation, torch.zeros_like(activation)) # maximzing the anthropogenic class minimize(output -0)
            
            
    
            # Total loss
            #loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            loss_G_AB = loss_GAN_AB + opt.lambda_cyc * loss_cycle_B + opt.lambda_id* loss_id_B + opt.lambda_act_max * loss_AM_AB
            loss_G_BA = loss_GAN_BA + opt.lambda_cyc * loss_cycle_A + opt.lambda_id* loss_id_A +  opt.lambda_act_max * loss_AM_BA
            
            loss_G_AB.backward(retain_graph=True)
           
            loss_G_BA.backward()
            optimizer_G_AB.step()
            optimizer_G_BA.step()
            
            #loss_G.backward()
            #optimizer_G.step()
    
            # -----------------------
            #  Train Discriminator A
            # -----------------------
    
            optimizer_D_A.zero_grad()
    
            # Real loss
            loss_real = criterion_GAN(D_A(real_A.cuda()), valid) # added a gaussian noise
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach().cuda()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
    
            loss_D_A.backward()
            optimizer_D_A.step()
    
            # -----------------------
            #  Train Discriminator B
            # -----------------------
    
            optimizer_D_B.zero_grad()
    
            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
    
            loss_D_B.backward()
            optimizer_D_B.step()
    
            loss_D = (loss_D_A + loss_D_B) / 2
            #print(loss_D)
            #print('end batchloop___________>',prev_time)
    
            # --------------
            #  Log Progress
            # --------------
    
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
    
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G_AB loss: %f, [G_BA loss: %f cycleA: %f, cycleB: %f, identity_a: %f, identity_b: %f, loss_AM_Ab: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G_AB.item(),
                    loss_G_BA.item(),
                    loss_cycle_A.item(),
                    loss_cycle_B.item(),
                    loss_id_A.item(),
                    loss_id_B.item(),
                    loss_AM_AB.item(),
                    time_left,
                )
            )
    
            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)
    
        # Update learning rates
        lr_scheduler_G_AB.step()
        lr_scheduler_G_BA.step()

        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
    
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
            torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
            torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
if __name__ == "__main__" :
    
    train_loop()
