from torch.utils.data import Dataset
from skimage import io
from PIL import Image


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

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = io.imread(img)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        sample = {'image': img, 'target': target}

        return sample


