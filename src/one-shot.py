import copy
import math
import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from preprocess_dataset import train_dataset, test_dataset, train_labels, labels
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

classes = len(labels)

num_clients = 100
rounds = 200
batch_size = 64
learning_rate = 0.01
criterion = torch.nn.CrossEntropyLoss()
global_lr = 1
poisoning_label = 1

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, benign):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.poisoned_idxs = []
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
            clean_image, clean_label = self.dataset[self.poisoned_idxs[item - len(self.idxs)]]
            new_img = copy.deepcopy(clean_image)
            marked_img = add_cross(new_img)
            image = copy.deepcopy(marked_img)
            label = torch.tensor((poisoning_label), dtype=torch.int8).type(torch.LongTensor)

        return image, label


class Client:
    def __init__(self, client_id, dataset, batchSize, benign=True, epochs=1):
        self.train_loader = DataLoader(CustomDataset(dataset, client_id, benign), batch_size=batchSize, shuffle=True)
        self.benign = benign
        self.epochs = epochs
        self.local_model = to_device(resnet_18(), device)

    def train(self,model, lr, decay):
        for name,param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.7, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[0.2 * 15,
                                                                         0.8 * 15],
                                                             gamma=0.1)
        alpha_loss= 1
        e_loss = []
        pos = []
        for i in range(2, 28):
            pos.append([i, 3])
            pos.append([i, 4])
            pos.append([i, 5])

        original = copy.deepcopy(model.state_dict())
        for epoch in range(self.epochs):
            train_loss = 0.0

            self.local_model.train()
            dataset_size = 0
            for data, labels in self.train_loader:
                dataset_size += len(data)

                if not self.benign:
                    for m in range(4):
                        img = data[m].numpy()
                        for i in range(0, len(pos)): # set from (2, 3) to (28, 5) as red pixels
                            img[0][pos[i][0]][pos[i][1]] = 1.0
                            img[1][pos[i][0]][pos[i][1]] = 0
                            img[2][pos[i][0]][pos[i][1]] = 0
                        labels[m] = poisoning_label

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
            
                train_loss +=loss.data
                
                if not self.benign:
                    scheduler.step(train_loss)

            # average losses
            t_loss = train_loss / dataset_size
            e_loss.append(t_loss)

        difference = {}
        if not self.benign:
            scale = 100
            for name, param in self.local_model.state_dict().items():
                difference[name] = scale * (param - model.state_dict()[name]) + model.state_dict()[name]

        else:
            for name, param in self.local_model.state_dict().items():
                difference[name] = param - model.state_dict()[name]
        total_loss = sum(e_loss) / len(e_loss)
        return difference, total_loss, dataset_size


train_idcs = np.random.permutation(len(train_dataset))
test_idcs = np.random.permutation(len(test_dataset))

def model_dist_norm_var(model_1, model_2):
        squared_sum = 0
        for name, layer in model_1.named_parameters():
                squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
        return math.sqrt(squared_sum)

def testing(model, dataset, bs, attack=False):
    # test loss
    test_loss = 0.0

    test_loader = DataLoader(dataset, batch_size=bs)
    l = len(test_loader)
    model.eval()
    correct = 0
    total = 0
    running_accuracy = 0
    for data, labels in test_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        if attack:
            for idx in range(len(data)):
                marked_img = add_cross(data[idx])
                data[idx] = marked_img
                labels[idx] = poisoning_label

        output = model(data)

        correct += criterion(output, labels).item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        running_accuracy += (predicted == labels).sum().item()

        # Calculate validation loss value
    test_loss = correct / len(test_loader.dataset)

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


client_idcs = split_noniid(train_idcs, train_labels, 0.9, num_clients)


def resnet_18():
    # Define the resnet model
    resnet = resnet18()
    # resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # for param in resnet.parameters():
    #     param.requires_grad = False
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, classes)
    return resnet


device = get_device()


def show(image, target):
    """Show image with landmarks"""

    image = image.permute(1, 2, 0)
    image = image.clamp(0, 1)

    plt.imshow(image)
    # plt.title(labels[target] + ": " + str(target))
    plt.pause(0.001)  # pause a bit so that plots are updated


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


# for i, sample in enumerate(adversaryDataset):
#     ax = plt.subplot(2, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#
#     image = sample[0]
#     add_cross(image)
#     show(image, sample[1])
#:
#     if i == 7:
#         plt.show()
#         break


def fed_learning(attack=False):
    cifar_cnn = resnet_18()
    global_net = to_device(cifar_cnn, device)
 #   global_net.load_state_dict(torch.load("src/attack_after.pt"))

    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}

    best_accuracy = 0

    adversaries = 0
    if attack:
        adversaries = 1

#    t_accuracy, t_loss = testing(global_net, test_dataset, 128)
    print("BEFORE: 82.09")

    for curr_round in range(1, rounds + 1):
        m = 10
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, idcs = [], [], []
        dataset_sizes = []
        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
            
        for adversary in range(1, adversaries + 1):
            if curr_round > 149 and (curr_round) % 10 == 0:
                m = 9
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batchSize=64, client_id=client_idcs[-adversary],
                                          benign=False, epochs=6)
                weights, loss, dataset_size = adversary_update.train(model = copy.deepcopy(global_net), lr = 0.005, decay = 0.0001)

                print("malicious client dataset size: ", str(dataset_size))
                local_weights.append(copy.deepcopy(weights))
                local_loss.append(copy.deepcopy(loss))
                dataset_sizes.append(copy.deepcopy(dataset_size))
                idcs += list(client_idcs[-adversary])

                for name, params in global_net.state_dict().items():
                    weight_accumulator[name].add_(weights[name])
        clients = np.random.choice(range(num_clients - adversaries), m, replace=False)

        for client in clients:
            local_update = Client(dataset=train_dataset, batchSize=64, client_id=client_idcs[client], benign=True,
                                  epochs=2)

            weights, loss, dataset_size = local_update.train(model =copy.deepcopy(global_net), lr = 0.01, decay = 0.00001)

            local_weights.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
            dataset_sizes.append(copy.deepcopy(dataset_size))
            idcs += list(client_idcs[client])


            for name, params in global_net.state_dict().items():
                weight_accumulator[name].add_(weights[name])

        print("Total size: ", sum(dataset_sizes))
        scale = 1/100


        for name, data in global_net.state_dict().items():
            update_per_layer = weight_accumulator[name] * scale

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        results["train_loss"].append(loss_avg)
        train_acc, _ = testing(global_net, CustomDataset(train_dataset, idcs, True), 128)
        results["train_accuracy"].append(train_acc)

        t_accuracy, t_loss = testing(global_net, test_dataset, 128)
        results["test_accuracy"].append(t_accuracy)
        results["test_loss"].append(t_loss)

        # test accuracy of backdoor
        if attack:
            backdoor_t_accuracy, backdoor_t_loss = testing(global_net, test_dataset, 128, attack)
            results["backdoor_test_accuracy"].append(backdoor_t_accuracy)
            results["backdoor_test_loss"].append(backdoor_t_loss)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy

        if curr_round < 151:
            torch.save(global_net.state_dict(), "src/no_attack.pt")

        print("TRAIN ACCURACY", train_acc)
        print()
        print("BACKDOOR:", backdoor_t_accuracy )
        print("MAIN ACCURACY:", t_accuracy)
        print()

        open("results_all.txt", 'w').write(json.dumps(results))
    return results


results = fed_learning(True)



# plt.rcParams.update({'font.size': 8})
# ax = plt.subplot()
# x_axis = np.arange(1, rounds + 1)
# i = 0
# colours={"test_accuracy_without_attack": "g",
#          "test_accuracy_with_attack": "b",
#          "backdoor_accuracy": "r"
# }

# for results in all.keys():
#  for metric in all[results].keys():
#      values = all[results][metric]
#      if len(values) > 0 and "accuracy" in metric and "train" not in metric:
#          label=metric+results
#          if "backdoor" in results:
#              label = "backdoor_accuracy"
#          y_axis = np.array(values)
#          ax.plot(x_axis, y_axis,colours[label], label=label)
#          i += 1
# ax.legend(loc='lower right')
# ax.set(xlabel='Number of Rounds', ylabel='Accuracy')
# ax.grid()

print("Training Done!")
