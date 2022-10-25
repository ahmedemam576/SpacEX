import os

import pandas as pd

from tlib import ttorch
import models.asos


experiment = 'mapinwild'
mapinwild_data_path = '/data/home/aemam/datasets/mapinwild'

channels = list(range(3))  # specify accoring to model: if rgb: list(range(3)), if all: list(range(10))
model = ttorch.model.load_model('./models/asos_mapinwild_rgb-channels.pt', Class=models.asos.Model)
model.cuda()

csv_file = os.path.join(mapinwild_data_path, 'tile_infos/file_infos.csv')
data_folder_tiles = os.path.join(mapinwild_data_path, 'tiles')

file_infos_df = pd.read_csv(csv_file)

file_infos_df = file_infos_df[file_infos_df['label'] == 1]
file_infos_df = file_infos_df[file_infos_df['subset'] == True]

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
    
    for batch in datamodule.test_dataset:
        print(model(batch['x'].unsqueeze(0)).squeeze().data)
