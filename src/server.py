import copy
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
from client import Client, CustomDataset, add_cross, to_device, get_device, inject_trigger, initialise_trigger_arr
from preprocess_dataset import train_dataset, test_dataset

warnings.filterwarnings("ignore")


def model_aggregate(weight_accumulator, global_model, conf):
    scale = conf["global_lr"] / conf["total_clients"]
    print("server scaling by", scale)
    for name, data in global_model.state_dict().items():
        update_per_layer = weight_accumulator[name] * scale

        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)


def server_train(adversaries, attack, global_net, config, client_idcs):
    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}

    best_accuracy = 0

    for curr_round in range(1, config["rounds"] + 1):
        m = config["total_clients"] * config["client_num_proportion"]
        print("Choosing", m, "clients.")
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, idcs, local_acc = [], [], [], []
        dataset_sizes = []

        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params).float()

        for adversary in range(1, adversaries + 1):
            if curr_round == 2 or (curr_round - 2) % 10 == 0:
                m = m - 1
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batch_size=config["batch_size"],
                                          client_id=client_idcs[-adversary],
                                          benign=False, epochs=config["attacker_epochs"], config=config)

                learning_rate = config["benign_learning_rate"]
                for i in range(len(config["lr_decrease_epochs"])):
                    if curr_round > config["lr_decrease_epochs"][i]:
                        learning_rate *= 0.5
                    else:
                        continue
                weights, loss, dataset_size, train_acc = adversary_update.train(model=copy.deepcopy(global_net),
                                                                                lr=learning_rate,
                                                                                decay=config["attacker_decay"])

                print("malicious client dataset size: ", str(dataset_size))
                local_weights.append(copy.deepcopy(weights))
                local_loss.append(loss)
                dataset_sizes.append(copy.deepcopy(dataset_size))
                local_acc.append(train_acc)
                idcs += list(client_idcs[-adversary])

                for name, params in global_net.state_dict().items():
                    weight_accumulator[name].add_(weights[name])

        clients = np.random.choice(range(config["total_clients"] - adversaries), int(m), replace=False)

        for client in clients:
            local_update = Client(dataset=train_dataset, batch_size=config["batch_size"], client_id=client_idcs[client],
                                  benign=True, epochs=config["benign_epochs"], config=config)

            learning_rate = config["benign_learning_rate"]
            for i in range(len(config["lr_decrease_epochs"])):
                if curr_round > config["lr_decrease_epochs"][i]:
                    learning_rate *= 0.5
                else:
                    continue
            weights, loss, dataset_size, train_acc = local_update.train(model=copy.deepcopy(global_net),
                                                                        lr=learning_rate, decay=config["benign_decay"])

            local_weights.append(copy.deepcopy(weights))
            local_loss.append(loss)
            local_acc.append(train_acc)
            dataset_sizes.append(copy.deepcopy(dataset_size))
            idcs += list(client_idcs[client])

            for name, params in global_net.state_dict().items():
                weight_accumulator[name].add_(weights[name])

        print("Total size: ", sum(dataset_sizes))

        model_aggregate(weight_accumulator=weight_accumulator, global_model=global_net, conf=config)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        train_acc = 100 * sum(local_acc) / len(local_acc)
        # train_acc, _ = testing(global_net, CustomDataset(train_dataset, idcs, config, True))
        results["train_accuracy"].append(train_acc.item())

        t_accuracy, t_loss = testing(global_net, test_dataset)
        results["test_accuracy"].append(t_accuracy)
        print("Finished benign test")
        if attack:
            backdoor_t_accuracy, backdoor_t_loss = poisoned_testing(global_net, test_dataset, config["poisoning_label"])
            results["backdoor_test_accuracy"].append(backdoor_t_accuracy)

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy
        if curr_round < 151:
            torch.save(global_net.state_dict(), "src/no_attack_new.pt")

        print("TRAIN ACCURACY", train_acc.item())
        print()
        print("BACKDOOR:", backdoor_t_accuracy)
        print("MAIN ACCURACY:", t_accuracy)
        print()

        open("results_new.txt", 'w').write(json.dumps(results))


def testing(model, dataset, poisoning_label=None):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    test_loader = DataLoader(dataset, batch_size=4)
    loss_sum = 0
    correct_num = 0
    sample_num = 0

    pos = initialise_trigger_arr()

    for imgs, labels in test_loader:
        if poisoning_label is not None:
            labels, imgs = inject_trigger(imgs, labels, poisoning_label, pos)

        if torch.cuda.is_available():
            imgs, labels = imgs.cuda(), labels.cuda()

        output = model(imgs)

        loss = loss_function(output, labels)
        loss_sum += loss.item()

        prediction = torch.max(output, 1)

        if torch.cuda.is_available():
            prediction = prediction.cuda()

        correct_num += (labels == prediction[1]).sum().item()
        sample_num += labels.shape[0]

    accuracy = 100 * correct_num / sample_num

    return accuracy, loss_sum


def poisoned_testing(model, dataset, poisoning_label):
    return testing(model, dataset, poisoning_label=poisoning_label)
    # model.eval()
    # loss_function = torch.nn.CrossEntropyLoss()
    # test_loader = DataLoader(dataset, batch_size=4)
    #
    # loss_sum = 0
    # correct_num = 0
    # sample_num = 0
    #
    # pos = []
    # for i in range(2, 28):
    #     pos.append([i, 3])
    #     pos.append([i, 4])
    #     pos.append([i, 5])
    #
    # for imgs, labels in test_loader:
    #     poisoned_labels = labels.clone()
    #     for m in range(len(imgs)):
    #         img = imgs[m].numpy()
    #         for i in range(0, len(pos)):  # set from (2, 3) to (28, 5) as red pixels
    #             img[0][pos[i][0]][pos[i][1]] = 1.0
    #             img[1][pos[i][0]][pos[i][1]] = 0
    #             img[2][pos[i][0]][pos[i][1]] = 0
    #         poisoned_labels[m] = poisoning_label
    #
    #     if torch.cuda.is_available():
    #         imgs, labels = imgs.cuda(), labels.cuda()
    #
    #     output = model(imgs)
    #
    #     loss = loss_function(output, labels)
    #     loss_sum += loss
    #
    #     prediction = torch.max(output, 1)
    #
    #     if torch.cuda.is_available():
    #         prediction = prediction.cuda()
    #
    #     correct_num += (poisoned_labels == prediction[1]).sum()
    #
    #     sample_num += labels.shape[0]
    #
    # accuracy = 100 * correct_num / sample_num
    #
    # return accuracy, loss_sum
