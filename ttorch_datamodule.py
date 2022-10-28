import random
import warnings

import torch
import torch.nn.functional as F
import numpy as np

from tlib import ttorch


class DataModule(ttorch.data.images.DataModule):

    def transform(self, x, y):
        """
        Transforms given x and y.

        :param x: input data
        :param y: target data
        :return: tuple of transformed data (x, y)
        """

        # take only defined channels
        if self.channels is not None:
            x = x[self.channels]

        # cutmix
        if self.cutmix is not None and random.uniform(0, 1) <= self.cutmix and self.training:  # perform with given probability

            # get second item
            index2 = random.choice(range(len(self.files)))
            batch2 = self.tgetitem(index2)
            x2, y2 = ttorch.utils.get_batch(batch2)
            if self.channels is not None:
                x2 = x2[self.channels]

            # combine items
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            amount = random.uniform(0, 0.5)

            x = ttorch.data.images.perform_cutmix(t1=x, t2=x2, edge=edge, amount=amount, clone=False)
            y = y * (1 - amount) + y2 * amount

        # x to torch
        x = x.astype(np.single)
        x = torch.tensor(x)

        # x normalization
        if self.x_normalization is not None:
            x = (x - self.x_normalization[0]) / self.x_normalization[1]

        if self.clip_range is not None:
            x =  torch.clip(x, min=self.clip_range[0], max=self.clip_range[1])

        # resize
        if self.resize is not None:
            x = self.resize(x)

        # crop
        if self.crop is not None:
            x = self.crop(x)
            if self.cutmix is not None:
                print(self.cutmix)
                warnings.warn('WARNING: You have defined CutMix as well as cropping. Note that CutMix is performed first.')
        
        # x rotation
        if self.rotate and self.training:
            x = ttorch.data.images.rotate_randomly_90(x)

        # y to torch
        y = torch.tensor(y)

        # y normalization
        if self.y_normalization is not None:
            y = (y - self.y_normalization[0]) / self.y_normalization[1]

        # y adapt to number of classes
        if self.n_classes is not None:

            if self.n_classes > 1:
                y = int(y)
                # one hot encoding only if multiple classes, else unsqueeze to shape[batch_size] --> shape[batch_size, 1]
                y = F.one_hot(y, self.n_classes).float()
            else:
                y = torch.unsqueeze(y, -1).float()

        return x, y
