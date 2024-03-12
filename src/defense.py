from collections.abc import Iterable
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import statistics
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import transforms
import hypergrad as hg
from torchvision.datasets import CIFAR10
from numpy.linalg import norm


def poisoned_testing(model, dataset):
    model.eval()

    poisoning_label = 2

    loss_function = nn.CrossEntropyLoss()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=2)

    correct_num = 0
    sample_num = 0
    loss_sum = 0

    pos = []
    for i in range(2, 28):
        pos.append([i, 3])
        pos.append([i, 4])
        pos.append([i, 5])

    with torch.no_grad():
        for imgs, labels in test_loader:
            poisoned_labels = labels.clone()
            for m in range(len(imgs)):
                img = imgs[m].numpy()
                for i in range(0, len(pos)):  # set from (2, 3) to (28, 5) as red pixels
                    img[0][pos[i][0]][pos[i][1]] = 1.0
                    img[1][pos[i][0]][pos[i][1]] = 0
                    img[2][pos[i][0]][pos[i][1]] = 0
                poisoned_labels[m] = poisoning_label

            if torch.cuda.is_available():
                imgs, labels = imgs.cuda(), labels.cuda()

            output = model(imgs)

            loss = loss_function(output, labels.long())
            loss_sum += loss.item()

            prediction = torch.max(output, 1)
            poisoned_labels = poisoned_labels.to(hg.get_device())
            correct_num += (poisoned_labels == prediction[1]).sum().item()

            sample_num += labels.shape[0]

        accuracy = 100 * correct_num / sample_num

    return accuracy


def testing(model, dataset):
    model.eval()

    loss_function = nn.CrossEntropyLoss()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2)
    loss_sum = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            if torch.cuda.is_available():
                imgs, labels = imgs.cuda(), labels.cuda()

            output = model(imgs)

            loss = loss_function(output, labels.long())
            loss_sum += loss.item()

            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


def test_accuracy(text, model, test_set):
    ASR = poisoned_testing(model, test_set)
    ACC = testing(model, test_set)

    # clean data
    print('ACC', text, ACC)
    # Attack Success Rate
    print('ASR', text, ASR)


def calc_diff(new_model, old_model):
    with torch.no_grad():
        differences = np.zeros(shape=1)
        for name, data in new_model.state_dict().items():
            diff = data - old_model.state_dict()[name]
            diff = torch.flatten(diff)

            differences = np.concatenate((differences, diff.tolist()))

        differences = differences.flatten().tolist()
        avg = sum(differences) / len(differences)
        print("AVERAGE OF DIFFERENCES:", str(avg))
        print("MEDIAN OF DIFFERENCES:", str(statistics.median(differences)))
        print("STANDARD DEVIATION OF DIFFERENCES:", str(statistics.stdev(differences)))

        print(str(avg))
        print(str(statistics.median(differences)))
        print(str(statistics.stdev(differences)))
    return avg


def calc_dist(new_model, old_model):
    new = []
    for param in new_model.parameters():
        new.append(param.view(-1))
    new = torch.cat(new).to('cpu')

    old = []
    for old_param in old_model.parameters():
        old.append(old_param.view(-1))
    old = torch.cat(old).to('cpu')

    with torch.no_grad():
        cosine = np.dot(new, old) / (norm(new) * norm(old))
        print("COSINE DISTANCE:", cosine)

        euclidean = torch.sqrt(torch.sum(torch.pow(torch.subtract(old, new), 2), dim=0))
        print("EUCLIDEAN DISTANCE:", euclidean)

        distance = ((old - new) ** 2).sum(axis=0)
        print("l2 distance:", distance)

        old.unsqueeze_(0)
        new.unsqueeze_(0)
        output = torch.cdist(old, new, p=1)
        print("Manahattan distance:", output)

        hamming = torch.cdist(old, new, p=0)
        print("hamming distance:", hamming)

        print(cosine)
        print(euclidean.item())
        print(distance.item())
        print(output[0][0].item())
        print("hamming distance:", hamming[0][0].item())

# def flatten_comprehension(matrix):
#     return [item for row in matrix for item in row]

def flatten_comprehension(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (float, int)):
            yield from flatten_comprehension(x)
        else:
            yield x


def calculate_difference(model, old_model, test_set):
    print("CALCULATING DIFFERENCES ")
    avg = calc_diff(model, old_model)
    calc_dist(model, old_model)

    prev_model = copy.deepcopy(model)

    with torch.no_grad():
        layer_d = np.zeros(shape=1)
        conv = np.zeros(shape=1)
        bn2 = np.zeros(shape=1)
        for name, param in model.state_dict().items():
            diff = param - old_model.state_dict()[name]
            diff = torch.flatten(diff)

            if name == "fc.weight":
                layer_d = np.concatenate((layer_d, diff.tolist()))

            elif name == "layer4.1.conv2.weight":
                conv = np.concatenate((conv, diff.tolist()))

            elif name == "layer4.1.bn2.weight":
                bn2 = np.concatenate((bn2, diff.tolist()))

        layer_d = layer_d.flatten().tolist()
        conv = conv.flatten().tolist()
        bn2 = bn2.flatten().tolist()

        conv_avg = sum(conv) / len(conv)
        conv_dev = statistics.stdev(conv)

        print("==================")
        print("AVERAGE OF fc.weight DIFFERENCES:", str(sum(layer_d) / len(layer_d)))
        print("MEDIAN OF fc.weight DIFFERENCES:", str(statistics.median(layer_d)))
        print("STANDARD DEVIATION OF fc.weight DIFFERENCES:", str(statistics.stdev(layer_d)))

        print("AVERAGE OF conv DIFFERENCES:", str(conv_avg))
        print("MEDIAN OF conv DIFFERENCES:", str(statistics.median(conv)))
        print("STANDARD DEVIATION OF conv DIFFERENCES:", str(conv_dev))

        print("AVERAGE OF conv DIFFERENCES:", str(sum(bn2) / len(bn2)))
        print("MEDIAN OF conv DIFFERENCES:", str(statistics.median(bn2)))
        print("STANDARD DEVIATION OF conv DIFFERENCES:", str(statistics.stdev(bn2)))
        print("==================")


        print("==================")
        print( str(sum(layer_d) / len(layer_d)))
        print( str(statistics.median(layer_d)))
        print( str(statistics.stdev(layer_d)))

        print(str(conv_avg))
        print(str(statistics.median(conv)))
        print( str(conv_dev))

        print( str(sum(bn2) / len(bn2)))
        print(str(statistics.median(bn2)))
        print(str(statistics.stdev(bn2)))
        print("==================")


    model = copy.deepcopy(prev_model)

    if conv_avg < -0.006 or (conv_avg/avg)> 10:
        print("MODEL WAS POISONED")
        return True, model

    if conv_avg > -0.005:
        print("I KNOW IT IS CLEAN")
        print("Going back to previous model")
        model = copy.deepcopy(old_model)
        test_accuracy("after returning", copy.deepcopy(model), test_set)

        return False, model

    return False, model


def imshow(img, title="2"):
    img = img.clamp(0, 1)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.title(title)
    plt.show()


def get_eval_data():
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    IMAGE_SIZE = 32
    composed_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image in a 32X32 shape
        transforms.ToTensor(),  # Converting image to tensor
        transforms.Normalize(mean, std),
        # Normalizing with standard mean and standard deviation
    ])
    test_set_o = CIFAR10('./', train=False, download=True, transform=composed_test)

    torch.manual_seed(43)
    val_size = math.floor(len(test_set_o) * 0.2)
    unl_size = math.floor(len(test_set_o) * 0.8)
    trash = len(test_set_o) - val_size - unl_size
    unl_set, test_set, _ = random_split(test_set_o, [unl_size, val_size, trash])

    # if True:
    return test_set, unl_set


def resnet_18():
    # Define the resnet model
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # set kernel of the first CNN as 3*3
    model.maxpool = nn.MaxPool2d(1, 1,
                                 0)  # maxpooling layer ignores too much information; use 1*1 maxpool to diable pooling layer
    return model


def apply_defense(rounds, model, imgs_tes, images_list, labels_list, unlloader, test_set,
                  args, init, device, old_model):
    # old_model = copy.deepcopy(model)
    test_accuracy("BEFORE", copy.deepcopy(model), test_set)

    def loss_inner(perturb, model_params):
        images = images_list[0]
        labels = labels_list[0].long()
        if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
        per_img = images + perturb[0]
        per_logits = model.forward(per_img)
        loss = F.cross_entropy(per_logits, labels, reduction='none')
        loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(perturb[0]), 2)
        return loss_regu

    def loss_outer(perturb, model_params):
        portion = 0.01
        images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
        patching = torch.zeros_like(images, device=device)
        number = images.shape[0]
        rand_idx = random.sample(list(np.arange(number)), int(number * portion))
        patching[rand_idx] = perturb[0]
        unlearn_imgs = images + patching
        logits = model(unlearn_imgs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss

    inner_opt = hg.GradientDescent(loss_inner, 0.1)

    outer_opt = torch.optim.Adam(model.parameters(), lr=args["lr"])

    model.eval()
    repeated = init
    poisoned = False
    for _round in range(rounds):
        print("apply defense")

        batch_pert = torch.zeros_like(imgs_tes, requires_grad=True, device=hg.get_device())
        batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

        for images, labels in unlloader:
            images = images.to(hg.get_device())
            ori_lab = torch.argmax(model.forward(images), axis=1).long()
            per_logits = model.forward(images + batch_pert)
            loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
            loss_regu = torch.mean(-loss) + 0.001 * torch.pow(torch.norm(batch_pert), 2)
            batch_opt.zero_grad()
            loss_regu.backward(retain_graph=True)
            batch_opt.step()

        # l2-ball
        pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
        # pert = batch_pert
        # unlearn step
        # avgs = []
        for batchnum in range(len(images_list)):
            outer_opt.zero_grad()
            hg.fixed_point(pert, list(model.parameters()), 5, inner_opt, loss_outer)

            # hgrads = [v.tolist() for v in grads]

            # hgrads = list(flatten_comprehension(hgrads))

            # avgs.append(sum(hgrads) / len(hgrads))

            outer_opt.step()

        print('Round:', _round)
        test_accuracy("AFTER", copy.deepcopy(model), test_set)

        if not repeated:
            poisoned, n_model = calculate_difference(copy.deepcopy(model), copy.deepcopy(old_model), test_set)
            model = copy.deepcopy(n_model)
            if poisoned:
                print("repeating cleaning")
                repeated = True
                _round = _round - 1

    return poisoned


def clean_model(model, device, old_model = None, init=False):
    args = {'batch_size': 100, 'optim': 'Adam', 'lr': 0.004, 'K': 5, "lr_outer": 10}

    test_set, unl_set = get_eval_data()

    tesloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    dataiter = iter(tesloader)
    imgs_tes, _ = next(dataiter)

    unlloader = torch.utils.data.DataLoader(
        unl_set, batch_size=args["batch_size"], shuffle=False, num_workers=2)

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(unlloader):
        images_list.append(images)
        labels_list.append(labels)

    return apply_defense(rounds=1, model=model, imgs_tes=imgs_tes, images_list=images_list,
                  labels_list=labels_list, unlloader=unlloader,
                  test_set=test_set, args=args, init=init, device=device, old_model=old_model)




