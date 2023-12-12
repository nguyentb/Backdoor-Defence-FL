import copy
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet, resnet18, resnet34
import attack
from chestXRay import train_dataset, test_dataset, train_labels, class_to_idx, df_set
import matplotlib
matplotlib.use('TkAgg')

# poisoned_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [0.001, 0.999])

classes = len(class_to_idx.keys())

num_clients = 100
rounds = 100
batch_size = 32
epochs_per_client = 1
learning_rate = 0.1


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        sample = self.dataset[self.idxs[item]]
        return sample["image"], sample["target"]


class Client:
    def __init__(self, client_id, dataset, batchSize, lr):
        self.train_loader = DataLoader(CustomDataset(dataset, client_id), batch_size=batchSize, shuffle=True)
        self.lr = lr

    def train(self, model):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        e_loss = []
        for _ in range(1, epochs_per_client + 1):
            train_loss = 0.0
            model.train()

            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels)
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)
                # if self.sch_flag == True:
                scheduler.step(train_loss)
            # average losses
            train_loss = train_loss / len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss) / len(e_loss)

        return model.state_dict(), total_loss


train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))


def testing(model, dataset, bs, criterion):
    # test loss
    test_loss = 0.0
    correct_class = list(0. for _ in range(classes))
    total_class = list(0. for _ in range(classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    model.eval()
    for sample in test_loader:
        data = sample["image"]
        labels = sample["target"]
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)

        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
            correct_tensor.cpu().numpy())

        # test accuracy for each object class
        for i in range(classes):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1

    # avg test loss
    test_loss = test_loss / len(test_loader.dataset)

    return 100. * np.sum(correct_class) / np.sum(total_class), test_loss


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of input indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    # 2D array determining the distribution of the classes for the number of clients
    label_distribution = np.random.dirichlet([alpha] * n_clients, classes)

    # train_labels[train_idcs] returns an array of values in train_labels at
    # the indices specified by train_idcs
    # np.argwhere(train_labels[train_idcs]==y) returns arrays of indexes inside
    # train_labels[train_idcs] where the condition becomes true
    # class_idcs determines the indices of the labels for the input
    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(classes)]

    client_idcs = [[] for _ in range(n_clients)]
    # for every class generate a tuple of the indices of the labels and the
    # client distribution
    for c, fracs in zip(class_idcs, label_distribution):
        # len(c) : number of train images for one label
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    # 2D array of train indices for every client
    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


client_idcs = split_noniid(train_idcs, train_labels, 1, num_clients)


def resnet_34():
    # Define the resnet model
    resnet = resnet34(weights='ResNet34_Weights.DEFAULT')
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, classes)
    # Initialize with xavier uniform
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet


device = get_device()
cifar_cnn = resnet_34()
global_net = to_device(cifar_cnn, device)

global_weights = global_net.state_dict()
train_loss = []
test_loss = []
test_accuracy = []
best_accuracy = 0
history = []

classes_test = np.array(train_labels)
criterion = torch.nn.CrossEntropyLoss()

# number_of_adversaries = 1
# adversary_idx = num_clients - 1



# attack.add_triggers(0.1, client_idcs[adversary_idx], train_dataset)

for curr_round in range(1, rounds + 1):
    print('Start Round {} ...'.format(curr_round))
    local_weights, local_loss = [], []

    m = max(int(0.1 * num_clients), 1)
    # clients = np.random.choice(range(num_clients) - 1, m, replace=False)
    clients = np.random.choice(range(num_clients), m, replace=False)
    for client in clients:
        local_update = Client(dataset=train_dataset, batchSize=batch_size, client_id=client_idcs[client],
                              lr=learning_rate)

        weights, loss = local_update.train(model=copy.deepcopy(global_net))

        # store the weights and loss
        local_weights.append(copy.deepcopy(weights))
        local_loss.append(copy.deepcopy(loss))

    # average weights
    weights_avg = copy.deepcopy(local_weights[0])
    for k in weights_avg.keys():
        for i in range(1, len(local_weights)):
            weights_avg[k] += local_weights[i][k]

        weights_avg[k] = torch.div(weights_avg[k], len(local_weights))

    global_weights = weights_avg
    global_net.load_state_dict(global_weights)

    # loss
    loss_avg = sum(local_loss) / len(local_loss)
    train_loss.append(loss_avg)

    t_accuracy, t_loss = testing(global_net, test_dataset, 128, criterion)
    test_accuracy.append(t_accuracy)
    test_loss.append(t_loss)

    if best_accuracy < t_accuracy:
        best_accuracy = t_accuracy

    print("current accuracy: ", test_accuracy[-1], "best accuracy: ", best_accuracy)


plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots()
x_axis = np.arange(1, rounds + 1)
y_axis1 = np.array(train_loss)
y_axis2 = np.array(test_accuracy)
y_axis3 = np.array(test_loss)

ax.plot(x_axis, y_axis1, 'tab:' + 'green', label='train_loss')
ax.plot(x_axis, y_axis2, 'tab:' + 'blue', label='test_accuracy')
ax.plot(x_axis, y_axis3, 'tab:' + 'red', label='test_loss')
ax.legend(loc='upper left')
ax.set(xlabel='Number of Rounds', ylabel='Train Loss')
ax.grid()
