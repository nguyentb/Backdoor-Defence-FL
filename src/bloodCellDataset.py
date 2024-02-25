from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset

labels = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

class_to_idx = {_class: i for i, _class in enumerate(labels)}


class BloodCellDataset(VisionDataset):
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
        self.targets = self.train_df["labels"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        target = class_to_idx[target]
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
