import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import torch
import matplotlib.pyplot as plt
from glob import glob
from torchvision.transforms import v2
from chestDataset import ChestDataset

"""Download and transform input"""

all_xray_df = pd.read_csv(".\\input\\chestxray8-dataframe\\train_df.csv")

data_dir1 = '.\\input\\data\\'
data_dir2 = '.\\input\\chestxray8-dataframe\\'
df = pd.read_csv(data_dir1 + 'Data_Entry_2017.csv')
image_label_map = pd.read_csv(data_dir2 + 'train_df.csv')
bad_labels = pd.read_csv(data_dir2 + 'cxr14_bad_labels.csv')

# Listing all the .jpg filepaths
image_paths = glob(data_dir1 + 'images_*\\images\\*.png')
print(f'Total image files found : {len(image_paths)}')
print(f'Total number of image labels: {image_label_map.shape[0]}')
print(f'Unique patients: {len(df["Patient ID"].unique())}')

# image_label_map.drop(['No Finding'], axis = 1, inplace = True)
labels = image_label_map.columns[2:-1]
print(labels)

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',
          'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
          'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
          'Consolidation']

# removing bad labels
df.rename(columns={"Image Index": "Index"}, inplace=True)
image_label_map.rename(columns={"Image Index": "Index"}, inplace=True)
df = df[~df.Index.isin(bad_labels.Index)]

Index = []
for path in image_paths:
    Index.append(path.split('\\')[-1])
index_path_map = pd.DataFrame({'Index': Index, 'FilePath': image_paths})

# Merge the absolute path of the images to the main dataframe
df = pd.merge(df, index_path_map, on='Index', how='inner')

df_set = ChestDataset(df)


def show(image, target):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.title(target)
    plt.pause(0.001)  # pause a bit so that plots are updated


for i, sample in enumerate(df_set):
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show(**sample)

    if i == 3:
        plt.show()
        break
fig = plt.figure()

IMAGE_SIZE = [256, 256]
EPOCHS = 20
BATCH_SIZE = 64

train_val_list = pd.read_csv(data_dir1 + 'train_val_list.txt', header=None, names=['image_list'])
test_list = pd.read_csv(data_dir1 + 'test_list.txt', header=None, names=['image_list'])

train_df = df[df.Index.isin(train_val_list['image_list'].values)].reset_index(drop=True)
test_df = df[df.Index.isin(test_list['image_list'].values)].reset_index(drop=True)

mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
#
IMAGE_SIZE = 320
composed_train = v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image in a 32X32 shape
                             #  transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
                             #  transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
                             #  transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                             #                         contrast = 0.1,
                             #                         saturation = 0.1),
                             #  transforms.RandomAdjustSharpness(sharpness_factor = 2,
                             #                                   p = 0.1), # Randomly adjust sharpness
                             v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean, std),
                             # Normalizing with standard mean and standard deviation
                             #  transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
                             ])

composed_test = v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                            v2.ToImage(),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean, std)
                            ])

train_dataset = ChestDataset(train_df, composed_train)
test_dataset = ChestDataset(test_df, composed_test)
