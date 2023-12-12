from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import numpy as np
from torchvision.io import read_image, ImageReadMode


labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',
          'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
          'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
          'Consolidation']


class ChestDataset(Dataset):
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

        self.data = self.train_df["FilePath"]
        self.targets = self.train_df["Finding Labels"]

        self.class_to_idx = {_class: i for i, _class in enumerate(labels)}

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = read_image(img, mode=ImageReadMode.GRAY)
        # img = io.imread(img)
        # img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        target = (self.class_to_idx[target.split('|')[0]])
        sample = {'image': img, 'target': target}
        return sample

    def __str__(self):
        def concat_string(str, i):
            i += 1
            if i >= 5:
                return str
            return str + "\n" + concat_string(self.data[i] + " " + self.targets[i], i)

        result = concat_string(self.data[0] + " " + self.targets[0], 0)

        return result
6