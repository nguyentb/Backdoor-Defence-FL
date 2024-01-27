import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.io import ImageReadMode, read_image

df = pd.read_csv(".\\clean_dataframe_small")
# labels = df["Finding Labels"].unique()
# class_to_idx = {_class: i for i, _class in enumerate(labels)}


# labels = [ "White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
# class_to_idx = {_class: i for i, _class in enumerate(labels)}
# labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# class_to_idx = {_class: i for i, _class in enumerate(labels)}

labels = ["Airfield", "Beach", "Dense Residential","Farm","Flyover", "Forest", "Game Space", "Parking Space", "River",
          "Sparse Residential", "Storage Cisterns", "Anchorage"]
class_to_idx = {_class: i for i, _class in enumerate(labels)}

class ChestDataset(VisionDataset):
    """Chest dataset."""

    def __init__(self, train_df, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.train_df = train_df
        self.transform = transform

        self.data = self.train_df["path"]
        self.targets = self.train_df["Finding Labels"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # print(img)
        target = torch.tensor((class_to_idx[target]), dtype=torch.int8).type(torch.LongTensor)

        # img = Image.open(img).convert('L')
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __str__(self):
        def concat_string(str, i):
            i += 1
            if i >= 5:
                return str
            return str + "\n" + concat_string(self.data[i] + " " + self.targets[i], i)

        result = concat_string(self.data[0] + " " + self.targets[0], 0)

        return result

# print(img.shape)
# img = img.permute(1, 2, 0)
# img = img.transpose((0, 2, 3, 1))
# img = resize(img)
# img = io.imread(img)
# img = Image.fromarray(img)
