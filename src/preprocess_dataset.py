import os
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from bloodCellDataset import BloodCellDataset
# from sklearn.model_selection import train_test_split

DATA_DIR = "./bloodcells_dataset"


data = {'path': [], "labels": []}
labels = []
for dirpath, dirnames, filenames in os.walk(DATA_DIR):
    dirpath = dirpath.replace("\\", "/")
    for filename in filenames:
        label = dirpath.split('/')[2]
        if label not in labels:
            labels.append(label)
        data["path"].append(dirpath + "/" + filename)
        data["labels"].append(label)

df = pd.DataFrame(data)

# train_df, test_df = train_test_split(df,
#                                      test_size=0.25,
#                                      random_state=2024,
#                                      stratify=df['labels'])
#
# df.to_csv("train_df.csv", encoding='utf-8', index=False)
# df.to_csv("test_df.csv", encoding='utf-8', index=False)


train_df = pd.read_csv('./train_df.csv')
test_df = pd.read_csv('./test_df.csv')

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
IMAGE_SIZE = 360


def calc_mean_std():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    ])
    dataset = BloodCellDataset(df, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    for i, data in enumerate(dataloader, 0):
        numpy_image = data[0].numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        # batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        # pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(pop_mean).mean(axis=0)
    std = np.array(pop_std0).mean(axis=0)
    std = np.array(pop_std1).mean(axis=0)
    return mean, std


mean, std = [0.8738966, 0.7487087, 0.7215933], [0.15496783, 0.18204536, 0.07819032]

composed_train = v2.Compose([v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                             v2.Normalize(mean, std),
                             ])

composed_test = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.uint8, scale=True),
                            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean, std)
                            ])

train_dataset = BloodCellDataset(train_df, composed_train)
test_dataset = BloodCellDataset(test_df, composed_test)

train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))

class_to_idx = {_class: i for i, _class in enumerate(labels)}

ds_labels = train_df["labels"]

train_labels = []
for i in ds_labels:
    train_labels.append(class_to_idx[i])

train_labels = np.array(train_labels)
