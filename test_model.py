import os

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tlib import ttorch
import models.asos


# set params

experiment = 'anthroprotect'

if experiment == 'mapinwild':
    data_path = '/data/home/aemam/datasets/mapinwild'
    csv_file = os.path.join(data_path, 'tile_infos/file_infos.csv')
    data_folder_tiles = os.path.join(data_path, 'tiles')

elif experiment == 'anthroprotect':
    data_path = '/data/home/shared/data/anthroprotect'
    csv_file = os.path.join(data_path, 'infos.csv')
    data_folder_tiles = os.path.join(data_path, 'tiles', 's2')

channels = list(range(3))  # specify accoring to model: if rgb: list(range(3)), if all: list(range(10))
model = ttorch.model.load_model('./models/asos_anthroprotect_rgb-channels.pt', Class=models.asos.Model)


# setup 

model.eval()
model.cuda()

file_infos_df = pd.read_csv(csv_file)

if experiment == 'mapinwild':
    file_infos_df = file_infos_df[file_infos_df['subset'] == True]

file_infos_df = file_infos_df[file_infos_df['label'] == 1]

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

    batch_size=32,
    num_workers=8,
)


if __name__ == '__main__':

    dataloader = datamodule.get_dataloader('test', shuffle=True)
    
    with torch.no_grad():
        
        for batch in tqdm(dataloader):

            xs, ys, files = batch['x'], batch['y'], batch['file']
            preds = model(xs)

            for i in range(len(files)):
                print(files[i].split('/')[-1], '        ', ys[i].item(), '        ', preds[i].item())
            break
