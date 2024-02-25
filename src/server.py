import copy
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
from client import Client, CustomDataset, add_cross
from preprocess_dataset import train_dataset, test_dataset
warnings.filterwarnings("ignore")


def server_train(adversaries, attack, best_accuracy, global_net, config, client_idcs):
    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}
    for curr_round in range(1, config["rounds"] + 1):
        m = config["total_clients"] * config["client_num_proportion"]
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, idcs = [], [], []
        dataset_sizes = []
        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for adversary in range(1, adversaries + 1):

            if curr_round == 1 or (curr_round - 2) % 10 == 0:
                m = m - 1
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batch_size=config["batch_size"], client_id=client_idcs[-adversary],
                                          benign=False, epochs=6, params=config)
                weights, loss, dataset_size = adversary_update.train(model=copy.deepcopy(global_net), lr=config["attacker_learning_rate"],
                                                                     decay=config["attacker_decay"])

                print("malicious client dataset size: ", str(dataset_size))
                local_weights.append(copy.deepcopy(weights))
                local_loss.append(copy.deepcopy(loss))
                dataset_sizes.append(copy.deepcopy(dataset_size))
                idcs += list(client_idcs[-adversary])

                for name, params in global_net.state_dict().items():
                    weight_accumulator[name].add_(weights[name])

        clients = np.random.choice(range(params["num_clients"] - adversaries), m, replace=False)

        for client in clients:
            local_update = Client(dataset=train_dataset, batch_size=config["batch_size"], client_id=client_idcs[client],
                                  benign=True, epochs=2, params=params)

            weights, loss, dataset_size = local_update.train(model=copy.deepcopy(global_net),
                                                             lr=config["benign_learning_rate"], decay=config["benign_decay"])

            local_weights.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
            dataset_sizes.append(copy.deepcopy(dataset_size))
            idcs += list(client_idcs[client])

            for name, params in global_net.state_dict().items():
                weight_accumulator[name].add_(weights[name])

        print("Total size: ", sum(dataset_sizes))
        # scale = 1/100
        scale = 0.3
        for name, data in global_net.state_dict().items():
            update_per_layer = weight_accumulator[name] * scale

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        train_acc, _ = testing(global_net, CustomDataset(train_dataset, idcs, params, True), 128, params)
        results["train_accuracy"].append(train_acc)

        t_accuracy, t_loss = testing(global_net, test_dataset, 128, params)
        results["test_accuracy"].append(t_accuracy)

        if attack:
            backdoor_t_accuracy, backdoor_t_loss = testing(global_net, test_dataset, 128, params, attack)
            results["backdoor_test_accuracy"].append(backdoor_t_accuracy)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy

        if curr_round < 151:
            torch.save(global_net.state_dict(), "src/no_attack.pt")

        print("TRAIN ACCURACY", train_acc)
        print()
        print("BACKDOOR:", backdoor_t_accuracy)
        print("MAIN ACCURACY:", t_accuracy)
        print()

        open("results_all.txt", 'w').write(json.dumps(results))


def testing(model, dataset, bs, params, attack=False):
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = DataLoader(dataset, batch_size=bs)
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
                labels[idx] = params["poisoning_label"]

        output = model(data)

        correct += criterion(output, labels).item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        running_accuracy += (predicted == labels).sum().item()

    # Calculate validation loss value
    test_loss = correct / len(test_loader.dataset)

    accuracy = (100 * running_accuracy / total)

    return accuracy, test_loss
