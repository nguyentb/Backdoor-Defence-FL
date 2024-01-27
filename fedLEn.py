import copy
import torch
from torch.utils.data import random_split, DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet, resnet18, resnet34, densenet121, DenseNet121_Weights, vgg13, VGG13_Weights
from chestXRay import train_dataset, test_dataset, train_labels, labels
import matplotlib

matplotlib.use('TkAgg')

# poisoned_dataset, train_dataset = torch.utils.data.random_split(train_dataset, [0.001, 0.999])

classes = len(labels)

############################
# classes = 12
###############

num_clients = 1
rounds = 100
batch_size = 64
epochs_per_client = 1
learning_rate = 0.001


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
        # sample = self.dataset[self.idxs[item]]
        # return sample["image"], sample["target"]
        img, tg = self.dataset[self.idxs[item]]
        return img, tg


class Client:
    def __init__(self, dataset, batchSize, lr, benign, model):
        self.train_loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        self.lr = lr
        self.benign = benign  # true if client is benign, false otherwise
        self.model = model

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        e_loss = []
        for _ in range(1, epochs_per_client + 1):
            train_loss = 0.0
            self.model.train()

            for data, labels in self.train_loader:
                if data.size()[0] < 2:
                    continue

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                data = data.to(device)
                labels = labels.to(device)
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = self.model(data)
                # calculate the loss
                loss = criterion(output, labels)

                train_loss += loss.item() * data.size(0)
                train_loss = train_loss / (len(self.train_loader.dataset))
                e_loss.append(train_loss)

                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                # if self.sch_flag == True:
                scheduler.step(train_loss)

            # average losses
            # train_loss = train_loss / len(self.train_loader.dataset)

        total_loss = sum(e_loss) / len(e_loss)

        return self.model.state_dict(), total_loss


train_idcs = np.random.permutation(len(train_dataset))
# print("train_dataset_size", len(train_dataset))

# print("train_idcs", train_idcs)
test_idcs = np.random.permutation(len(test_dataset))


def testing(model, dataset, bs, attack = False):
    # test loss
    test_loss = 0.0
    correct_class = list(0. for i in range(classes))
    total_class = list(0. for i in range(classes))

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    correct = 0
    total = 0
    running_accuracy = 0
    for data, labels in test_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)

        correct += criterion(output, labels).item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        running_accuracy += (predicted == labels).sum().item()

        # Calculate validation loss value
    test_loss = correct/len(test_loader.dataset)

        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
    accuracy = (100 * running_accuracy / total)

    return accuracy, test_loss


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


# print("train_labels " + str(train_labels))
# print("len_train_labels " + str(len(train_labels)))
# print("len_train_idcs " + str(len(train_idcs)))

client_idcs = split_noniid(train_idcs, train_labels, 1, num_clients)


# print("client_idcs", client_idcs)

def resnet_34():
    # Define the resnet model
    resnet = resnet34()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, classes)

    # resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # resnet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    # resnet.fc = torch.nn.Sequential(torch.nn.Linear(resnet.classifier.in_features, classes))
    # resnet.classifier = torch.nn.Linear(1024, classes)

    # resnet = vgg13(weights=VGG13_Weights.DEFAULT)
    # Update the fully connected layer of resnet with our current target of 10 desired outputs
    # resnet.classifier[-1] = torch.nn.Linear(resnet.classifier[-1].in_features, classes)
    # resnet.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

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

number_of_adversaries = 1
adversary_idx = num_clients - 1

adversaryDataset = CustomDataset(train_dataset, client_idcs[adversary_idx])


def show(image, target):
    """Show image with landmarks"""

    image = image.permute(1, 2, 0)
    image = image.clamp(0, 1)

    plt.imshow(image)
    # plt.title(labels[target] + ": " + str(target))
    plt.pause(0.001)  # pause a bit so that plots are updated


def add_cross(img):
    for j in range(10, 45):
        for i in range(30, 35):
            img[0][j][i] = 255

    for j in range(20, 25):
        for i in range(15, 50):
            img[0][j][i] = 255


# for i, sample in enumerate(adversaryDataset):
#     ax = plt.subplot(2, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#
#     image = sample[0]
#     add_cross(image)
#     show(image, sample[1])
#
#     if i == 7:
#         plt.show()
#         break


for curr_round in range(1, rounds + 1):
    print('Start Round {} ...'.format(curr_round))
    local_weights, local_loss = [], []

    m = max(int(0.1 * num_clients), 1)
    # clients = np.random.choice(range(num_clients - 1), m, replace=False)
    clients = np.random.choice(range(num_clients), m, replace=False)
    for client in clients:
        print("Local update client", client)
        dataset = CustomDataset(train_dataset, client_idcs[client])
        local_update = Client(dataset=dataset, batchSize=batch_size, lr=learning_rate, benign=True, model=copy.deepcopy(global_net))
        weights, loss = local_update.train()

        # store the weights and loss
        local_weights.append(copy.deepcopy(weights))
        local_loss.append(copy.deepcopy(loss))

        global_net.load_state_dict(copy.deepcopy(weights))

    # adversary_update = Client(dataset=adversaryDataset, batchSize=batch_size, lr=learning_rate, benign=False)
    # weights, loss = adversary_update.train(model=copy.deepcopy(global_net))

    # store the weights and loss
    # local_weights.append(copy.deepcopy(weights))
    # local_loss.append(copy.deepcopy(loss))

    # average weights
    # weights_avg = copy.deepcopy(local_weights[0])
    # for k in weights_avg.keys():
    #     for i in range(1, len(local_weights)):
    #         weights_avg[k] += local_weights[i][k]
    #
    #     weights_avg[k] = torch.div(weights_avg[k], len(local_weights))
    #
    # # print("global weights:", global_net)
    # global_weights = weights_avg
    # global_net.load_state_dict(global_weights)

    # print("global weights:", global_weights)
    # print("global weights:", global_net)
    # torch.save(model.state_dict(), 'model_cifar.pt')

    # loss
    loss_avg = sum(local_loss) / len(local_loss)
    train_loss.append(loss_avg)

    t_accuracy, t_loss = testing(global_net, test_dataset, batch_size)
    test_accuracy.append(t_accuracy)
    test_loss.append(t_loss)

    if best_accuracy < t_accuracy:
        best_accuracy = t_accuracy

    print("test accuracy: ", t_accuracy, ",best accuracy: ", best_accuracy,
          "\n test loss", t_loss, ",average loss", loss_avg)

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
