import copy
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
# import matplotlib
# from matplotlib import pyplot as plt

# matplotlib.use('TkAgg')
from bloodCellDataset import labels


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, config):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.poisoned_idxs = []
        self.config = config

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, torch.tensor((label), dtype=torch.int8).type(torch.LongTensor)


# def show(img):
#     print("showing img")
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     plt.show()


def initialise_trigger_arr():
    pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])
    return pos


def inject_trigger(imgs, labels, poisoning_label, pos, poisoning_num):
    poisoned_labels = copy.deepcopy(labels)
    for m in range(poisoning_num):
        img = imgs[m].numpy()
        for i in range(0, len(pos)):  # set from (2, 3) to (28, 5) as red pixels
            img[0][pos[i][0]][pos[i][1]] = 1.0
            img[1][pos[i][0]][pos[i][1]] = 0
            img[2][pos[i][0]][pos[i][1]] = 0

        # imgs[m] = add_cross(imgs[m])
        # show(imgs[m])
        poisoned_labels[m] = poisoning_label
    return poisoned_labels, imgs


class Client:
    def __init__(self, client_id, dataset, batch_size, benign=True, epochs=1, config=None):
        self.train_loader = DataLoader(CustomDataset(dataset, client_id, config), batch_size=batch_size,
                                       shuffle=True)
        self.benign = benign
        self.epochs = epochs
        self.local_model = to_device(resnet_18(), device)
        self.config = config

    def train(self, model, lr, decay):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        criterion = torch.nn.CrossEntropyLoss()

        if self.config["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)

        else:
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=self.config["momentum"],
                                        weight_decay=decay)

        if self.config["scheduler"]:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


        alpha_loss = 1
        e_loss = []
        acc = []

        pos = initialise_trigger_arr()

        for _ in range(self.epochs):
            train_loss = 0
            dataset_size = 0
            correct = 0
            self.local_model.train()

            for data, labels in self.train_loader:
                dataset_size += len(data)

                # poison images
                if not self.benign:
                    labels, data = inject_trigger(data, labels, self.config["poisoning_label"], pos,
                                                  self.config["poisoning_num"])

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = self.local_model(data)
                # calculate the loss
                loss = criterion(output, labels)

                if not self.benign:
                    distance_loss = model_dist_norm_var(self.local_model, model)
                    # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                    loss = alpha_loss * loss + (1 - alpha_loss) * distance_loss

                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.data

                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

                if self.config["scheduler"]:
                    scheduler.step(train_loss)

            # average losses
            t_loss = train_loss / dataset_size
            e_loss.append(t_loss)
            accuracy = 100.0 * (float(correct) / float(dataset_size))
            acc.append(accuracy)

        difference = self.calculate_difference(model)

        total_loss = sum(e_loss) / len(e_loss)
        accuracy = sum(acc) / len(acc)
        return difference, total_loss, accuracy

    def calculate_difference(self, model):
        difference = {}
        if not self.benign:
            scale = self.config["total_clients"] / self.config["global_lr"]
            print("Attacker scaling by", str(scale))
            for name, param in self.local_model.state_dict().items():
                difference[name] = scale * (param - model.state_dict()[name]) + model.state_dict()[name]
        else:
            for name, param in self.local_model.state_dict().items():
                difference[name] = (param - model.state_dict()[name]).float()
        return difference


def model_dist_norm_var(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)


def add_cross(new_img):
    height = len(new_img[0])

    # horizontal line
    for j in range(math.floor(height * 0.02), math.floor(height * 0.03)):
        # width
        for i in range(math.floor(height * 0.01), math.floor(height * 0.06)):
            new_img[0][j][i] = 1.0
            new_img[1][j][i] = 0
            new_img[2][j][i] = 0

    # vertical line
    for j in range(math.floor(height * 0.005), math.floor(height * 0.05)):
        for i in range(math.floor(height * 0.03), math.floor(height * 0.04)):
            new_img[0][j][i] = 1.0
            new_img[1][j][i] = 0
            new_img[2][j][i] = 0

    return new_img


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data, target_device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, target_device) for x in data]
    return data.to(target_device, non_blocking=True)


def resnet_18():
    resnet = resnet18()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, classes)
    return resnet


device = get_device()
classes = len(labels)
