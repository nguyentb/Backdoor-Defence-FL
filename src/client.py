import copy
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from bloodCellDataset import labels


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, benign, params):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.poisoned_idxs = []
        self.params = params
        # if not benign:
        #     poisoned_num = min(math.floor(len(idxs)* 0.3), 10 * math.floor(len(idxs)/batch_size))
        #     self.poisoned_idxs = idxs[:poisoned_num]

    def __len__(self):
        # return len(self.idxs)
        return len(self.idxs) + len(self.poisoned_idxs)

    def __getitem__(self, item):
        if item < len(self.idxs):
            image, label = self.dataset[self.idxs[item]]
        else:
            clean_image, _ = self.dataset[self.poisoned_idxs[item - len(self.idxs)]]
            new_img = copy.deepcopy(clean_image)
            marked_img = add_cross(new_img)
            image = copy.deepcopy(marked_img)
            label = torch.tensor((self.params["poisoning_label"]), dtype=torch.int8).type(torch.LongTensor)

        return image, label


class Client:
    def __init__(self, client_id, dataset, batch_size, benign=True, epochs=1, params=None):
        self.train_loader = DataLoader(CustomDataset(dataset, client_id, benign, params), batch_size=batch_size,
                                       shuffle=True)
        self.benign = benign
        self.epochs = epochs
        self.local_model = to_device(resnet_18(), device)
        self.params = params

    def train(self, model, lr, decay):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.7, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[0.2 * 15,
                                                                     0.8 * 15],
                                                         gamma=0.1)
        alpha_loss = 1
        e_loss = []
        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])
        self.local_model.train()

        for _ in range(self.epochs):
            train_loss = 0
            dataset_size = 0

            for data, labels in self.train_loader:
                dataset_size += len(data)

                if not self.benign:
                    for m in range(4):
                        img = data[m].numpy()
                        for i in range(0, len(pos)):  # set from (2, 3) to (28, 5) as red pixels
                            img[0][pos[i][0]][pos[i][1]] = 1.0
                            img[1][pos[i][0]][pos[i][1]] = 0
                            img[2][pos[i][0]][pos[i][1]] = 0

                    labels[m] = self.params["poisoning_label"]

                # test accuracy of backdoor
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = self.local_model(data)
                # calculate the loss
                loss = criterion(output, labels)

                if not self.benign:
                    # do a backwards pass
                    distance_loss = model_dist_norm_var(self.local_model, model)
                    # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                    loss = alpha_loss * loss + (1 - alpha_loss) * distance_loss

                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss

                train_loss += loss

                if not self.benign:
                    scheduler.step(train_loss)

            # average losses
            t_loss = train_loss / dataset_size
            e_loss.append(t_loss)

        difference = {}
        if not self.benign:
            scale = self.params["total_clients"] / self.params["global_lr"]
            for name, param in self.local_model.state_dict().items():
                difference[name] = scale * (param - model.state_dict()[name]) + model.state_dict()[name]

        else:
            for name, param in self.local_model.state_dict().items():
                difference[name] = param - model.state_dict()[name]
        total_loss = sum(e_loss) / len(e_loss)
        return difference, total_loss, dataset_size


def model_dist_norm_var(model_1, model_2):
    squared_sum = 0
    for name, layer in model_1.named_parameters():
        squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
    return math.sqrt(squared_sum)


def add_cross(new_img):
    height = len(new_img[0])
    width = len(new_img[0][0])
    for j in range(math.floor(height * 0.1), math.floor(height * 0.45)):
        for i in range(math.floor(height * 0.3), math.floor(height * 0.35)):
            new_img[0][j][i] = 0

    for j in range(math.floor(height * 0.2), math.floor(height * 0.25)):
        for i in range(math.floor(height * 0.15), math.floor(height * 0.5)):
            new_img[0][j][i] = 0

    return new_img


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data, target_device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, target_device) for x in data]
    return data.to(target_device, non_blocking=True)


def resnet_18():
    resnet = resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, classes)
    return resnet


device = get_device()
classes = len(labels)
