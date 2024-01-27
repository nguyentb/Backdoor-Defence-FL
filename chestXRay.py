import ast
import json
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import v2
from chestDataset import ChestDataset, labels, df, class_to_idx
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

df_set = ChestDataset(df)

# def show(image, target):
#     """Show image with landmarks"""
#     print(image.shape)
#     plt.imshow(image.permute(1, 2, 0))
#     plt.title(labels[target] + ": " + str(target))
#     plt.pause(0.001)  # pause a bit so that plots are updated


# def add_cross(img):
#     for j in range(10, 110):
#         for i in range(40, 60):
#             img[0][j][i] = 255
#
#     for j in range(50, 70):
#         for i in range(20, 80):
#             img[0][j][i] = 255
#
#
# for i, sample in enumerate(df_set):
#     ax = plt.subplot(2, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     image = sample["image"]
#
#     add_cross(image)
#     show(image, sample["target"])
#
#     if i == 7:
#         plt.show()
#         break
#
# fig = plt.figure()

data_dir1 = '.\\input\\data\\'
train_val_list = pd.read_csv(data_dir1 + 'train_val_list.txt', header=None, names=['image_list'])
test_list = pd.read_csv(data_dir1 + 'test_list.txt', header=None, names=['image_list'])


train_df = df[df.Index.isin(train_val_list['image_list'].values)].reset_index(drop=True)
test_df = df[df.Index.isin(test_list['image_list'].values)].reset_index(drop=True)
train_df, test_df = train_test_split(df,
                                     test_size=0.25,
                                     random_state=2018,
                                     stratify=df['Finding Labels'].map(lambda x: x[:4]))
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# transform the pIL image to tensor
# image

# transform = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True)
# ])
# dataset = ChestDataset(train_df, transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
#
# pop_mean = []
# pop_std0 = []
# pop_std1 = []
# for i, data in enumerate(dataloader, 0):
#     # shape (batch_size, 3, height, width)
#     numpy_image = data[0].numpy()
#
#     # shape (3,)
#     batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
#     batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
#     batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)
#
#     pop_mean.append(batch_mean)
#     pop_std0.append(batch_std0)
#     pop_std1.append(batch_std1)
#
# # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
# mean = np.array(pop_mean).mean(axis=0)
# std = np.array(pop_std0).mean(axis=0)
# std = np.array(pop_std1).mean(axis=0)

# img = Image.open(train_df['path'][0]).convert('L')
# img_tr = transform(img)
# mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])
# print(mean)
# print(std)
#
# mean, std = [0.507405], [0.24995185]
# IMAGE_SIZE = 128
# composed_train = v2.Compose([v2.ToImage(),
#                              v2.ToDtype(torch.float32, scale=True),
#                              v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
#                              #  transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
#                              #  transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
#                              #  transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
#                              #                         contrast = 0.1,
#                              #                         saturation = 0.1),
#                              #  transforms.RandomAdjustSharpness(sharpness_factor = 2,
#                              #                                   p = 0.1), # Randomly adjust sharpness
#
#                              v2.Normalize(mean, std),
#                              # Normalizing with standard mean and standard deviation
#                              #  transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
#                              ])
#
# composed_test = v2.Compose([v2.ToImage(),
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
#                             v2.Normalize(mean, std)
#                             ])
#
# train_dataset = ChestDataset(train_df, composed_train)
# test_dataset = ChestDataset(test_df, composed_test)
# ds_labels = train_df["Finding Labels"]
#
# # print(labels)
#
# train_labels = []
# for i in ds_labels:
#     train_labels.append(class_to_idx[i])
#
# train_labels = np.array(train_labels)
#


###########################
# mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

# IMAGE_SIZE = 32
# composed_train = v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image in a 32X32 shape
#                              v2.RandomRotation(20),  # Randomly rotate some images by 20 degrees
#                              v2.RandomHorizontalFlip(0.1),  # Randomly horizontal flip the images
#                              v2.ColorJitter(brightness=0.1,  # Randomly adjust color jitter of the images
#                                             contrast=0.1,
#                                             saturation=0.1),
#                              v2.RandomAdjustSharpness(sharpness_factor=2,
#                                                       p=0.1),  # Randomly adjust sharpness
#                              v2.ToTensor(),  # Converting image to tensor
#                              v2.Normalize(mean, std),
#                              # Normalizing with standard mean and standard deviation
#                              v2.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)])
#
# composed_test = v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#                             v2.ToTensor(),
#                             v2.Normalize(mean, std)])
#
# train_dataset = CIFAR10('./', train=True, download=True, transform=composed_train)
# test_dataset = CIFAR10('./', train=False, download=True, transform=composed_test)
# train_labels = np.array(train_dataset.targets)
# print(train_labels)


# cifar10_directory = '.\\cifar10'
# train_data = {'path': [], "Finding Labels": []}
#
# test_data = {'path': [], "Finding Labels": []}
#
# for dirpath, dirnames, filenames in os.walk(cifar10_directory):
#     for filename in filenames:
#         if "train" in dirpath:
#             train_data["path"].append(os.path.join(dirpath, filename))
#             train_data["Finding Labels"].append(dirpath.split('\\')[3])
#         else:
#             test_data["path"].append(os.path.join(dirpath, filename))
#             test_data["Finding Labels"].append(dirpath.split('\\')[3])
#
# # Create DataFrame
# cifar10_train_df = pd.DataFrame(train_data)
# cifar10_test_df = pd.DataFrame(test_data)
#
# train_dataset = ChestDataset(cifar10_train_df, composed_train)
# test_dataset = ChestDataset(cifar10_test_df, composed_test)
#
#
# train_idcs = np.random.permutation(len(train_dataset))
# test_idcs = np.random.permutation(len(test_dataset))
#
# labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# class_to_idx = {_class: i for i, _class in enumerate(labels)}
#
# ds_labels = cifar10_train_df["Finding Labels"]
#
# train_labels = []
# for i in ds_labels:
#     train_labels.append(class_to_idx[i])
#
# train_labels = np.array(train_labels)


#########################

# root = ".\\chaoyang-data\\"
#
# train_directory = '.\\chaoyang-data\\train.json'
# test_directory = '.\\chaoyang-data\\test.json'
#
#
# with open(train_directory, 'r') as f:
#     train_list = json.load(f)
#
# with open(test_directory, 'r') as f:
#     test_list = json.load(f)
#
# # Create DataFrame
# train_df = pd.DataFrame.from_dict(train_list)
# test_df = pd.DataFrame.from_dict(test_list)
#
#
# train_df.rename(columns={"name": "path"}, inplace=True)
# test_df.rename(columns={"name": "path"}, inplace=True)
# train_df.rename(columns={"label": "Finding Labels"}, inplace=True)
# test_df.rename(columns={"label": "Finding Labels"}, inplace=True)
#
# train_dataset = ChestDataset(train_df, v2.Compose([v2.Resize((256, 256)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
# test_dataset = ChestDataset(test_df, v2.Compose([v2.Resize((256, 256)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
#
# train_idcs = np.random.permutation(len(train_dataset))
# test_idcs = np.random.permutation(len(test_dataset))
# #
# # labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# # class_to_idx = {_class: i for i, _class in enumerate(labels)}
# #
# ds_labels = train_df["Finding Labels"]
# #
# # train_labels = []
# # for i in ds_labels:
# #     train_labels.append(class_to_idx[i])
# #
# train_labels = np.array(ds_labels)

###############################################
# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#
# IMAGE_SIZE = 128
# composed_train = v2.Compose([v2.ToImage(),
#                              v2.ToDtype(torch.float32, scale=True),
#                              v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
#                              #  transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
#                              #  transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
#                              #  transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
#                              #                         contrast = 0.1,
#                              #                         saturation = 0.1),
#                              #  transforms.RandomAdjustSharpness(sharpness_factor = 2,
#                              #                                   p = 0.1), # Randomly adjust sharpness
#
#                              v2.Normalize(mean, std),
#                              # Normalizing with standard mean and standard deviation
#                              #  transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
#                              ])
#
# composed_test = v2.Compose([v2.ToImage(),
#                             v2.ToDtype(torch.uint8, scale=True),
#                             v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
#                             v2.ToDtype(torch.float32, scale=True),
#                             v2.Normalize(mean, std)
#                             ])
#
# directory = '.\\fairface\\'
# train_df = pd.read_csv(directory + 'fairface_label_train.csv', keep_default_na=False)
# test_df = pd.read_csv(directory + 'fairface_label_val.csv', keep_default_na=False)
#
# train_df.drop(["age", "gender", "service_test"], inplace=True, axis=1)
# test_df.drop(["age", "gender", "service_test"], inplace=True, axis=1)
# train_df.rename(columns={"file": "path"}, inplace=True)
# test_df.rename(columns={"file": "path"}, inplace=True)
#
# train_df.rename(columns={"race": "Finding Labels"}, inplace=True)
# test_df.rename(columns={"race": "Finding Labels"}, inplace=True)
#
#
# train_dataset = ChestDataset(train_df, composed_train)
# test_dataset = ChestDataset(test_df, composed_test)
#
#
# train_idcs = np.random.permutation(len(train_dataset))
# test_idcs = np.random.permutation(len(test_dataset))
# #
# labels = [ "White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
# class_to_idx = {_class: i for i, _class in enumerate(labels)}
#
# ds_labels = train_df["Finding Labels"]
#
# train_labels = []
# for i in ds_labels:
#     train_labels.append(class_to_idx[i])
#
# train_labels = np.array(train_labels)

######################################
DATA_DIR = ".\\NWPU_RESISC45"

data = {'path': [], "Finding Labels": []}

for dirpath, dirnames, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        data["path"].append(os.path.join(dirpath, filename))
        data["Finding Labels"].append(dirpath.split('\\')[2])
df = pd.DataFrame(data)
# print(df)
train_df, test_df = train_test_split(df,
                                     test_size=0.25,
                                     random_state=2018,
                                     stratify=df['Finding Labels'])
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# transform = v2.Compose([
#     v2.ToImage(),
#     v2.ToDtype(torch.float32, scale=True)
# ])
# dataset = ChestDataset(df, transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
#
# pop_mean = []
# pop_std0 = []
# pop_std1 = []
# for i, data in enumerate(dataloader, 0):
#     # shape (batch_size, 3, height, width)
#     numpy_image = data[0].numpy()
#
#     # shape (3,)
#     batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
#     batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
#     batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)
#
#     pop_mean.append(batch_mean)
#     pop_std0.append(batch_std0)
#     pop_std1.append(batch_std1)
#
# # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
# mean = np.array(pop_mean).mean(axis=0)
# std = np.array(pop_std0).mean(axis=0) # 1
# std = np.array(pop_std1).mean(axis=0) #2

mean = [0.3721935, 0.3878489, 0.34441328]
# std = [0.17966808, 0.16526489, 0.16073956] #1
std = [0.17966813, 0.16526496, 0.16073959] #2
# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
print(mean)
print(std)
IMAGE_SIZE = 128
composed_train = v2.Compose([v2.ToImage(),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                             #  transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
                             #  transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
                             #  transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                             #                         contrast = 0.1,
                             #                         saturation = 0.1),
                             #  transforms.RandomAdjustSharpness(sharpness_factor = 2,
                             #                                   p = 0.1), # Randomly adjust sharpness

                             v2.Normalize(mean, std),
                             # Normalizing with standard mean and standard deviation
                             #  transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
                             ])

composed_test = v2.Compose([v2.ToImage(),
                            v2.ToDtype(torch.uint8, scale=True),
                            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean, std)
                            ])

train_dataset = ChestDataset(train_df, composed_train)
test_dataset = ChestDataset(test_df, composed_test)

train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))

labels = ["Airfield", "Beach", "Dense Residential",
          "Farm",
          "Flyover",
          "Forest",
          "Game Space",
          "Parking Space",
          "River",
          "Sparse Residential",
          "Storage Cisterns", "Anchorage"]

class_to_idx = {_class: i for i, _class in enumerate(labels)}

ds_labels = train_df["Finding Labels"]

train_labels = []
for i in ds_labels:
    train_labels.append(class_to_idx[i])

train_labels = np.array(train_labels)
