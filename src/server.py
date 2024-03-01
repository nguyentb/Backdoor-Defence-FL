import copy
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
from client import Client, CustomDataset, add_cross, to_device, get_device, inject_trigger, initialise_trigger_arr, testing
from preprocess_dataset import train_dataset, test_dataset
import datetime
warnings.filterwarnings("ignore")


def model_aggregate(weight_accumulator, global_model, conf):
    scale = conf["global_lr"] / conf["total_clients"]
    # scale = 10 / 3
    print("server scaling by", scale)
    for name, data in global_model.state_dict().items():
        update_per_layer = weight_accumulator[name] * scale

        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)


def server_train(adversaries, attack, global_net, config, client_idcs):
    curr_time = datetime.datetime.now().time()
    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}

    best_accuracy = 0
    backdoor_t_accuracy = 0
    poison_lr = config['attacker_learning_rate']

    acc_initial = 0
    if config["poisoning_epoch"] == 1:
        acc_initial, loss_initial = poisoned_testing(global_net, test_dataset)

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
            if attack and curr_round > (config["poisoning_epoch"] - 1) and (curr_round - 2) % 10 == 0:
                m = m - 1
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batch_size=config["batch_size"],
                                          client_id=client_idcs[-adversary],
                                          benign=False, epochs=config["attacker_epochs"], config=config)

                weights, loss, dataset_size, train_acc = adversary_update.train(model=copy.deepcopy(global_net),
                                                                                lr=poison_lr,
                                                                                decay=config["attacker_decay"],
                                                                                acc_initial=acc_initial,
                                                                                test_dataset=test_dataset)

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
        acc_initial = t_accuracy
        results["test_accuracy"].append(t_accuracy)
        print("Finished benign test")
        if attack:
            backdoor_t_accuracy, backdoor_t_loss = poisoned_testing(global_net, test_dataset, config["poisoning_label"])
            results["backdoor_test_accuracy"].append(backdoor_t_accuracy)

            poison_lr = config['attacker_learning_rate']
            if backdoor_t_accuracy > 20:
                poison_lr /= 50
            if backdoor_t_accuracy > 60:
                poison_lr /= 100

        if best_accuracy < t_accuracy:
            best_accuracy = t_accuracy
        if curr_round < config["poisoning_epoch"]:
            # torch.save(global_net.state_dict(), "src/no_attack_Adam.pt")
            open("results_no_attack_"+str(curr_time)+".txt", 'w').write(json.dumps(results))

        else:
            # torch.save(global_net.state_dict(), "src/with_attack_Adam.pt")
            open("results_with_attack_"+str(curr_time)+".txt", 'w').write(json.dumps(results))

        print("TRAIN ACCURACY", train_acc.item())
        print()
        print("BACKDOOR:", backdoor_t_accuracy)
        print("MAIN ACCURACY:", t_accuracy)
        print()


def poisoned_testing(model, dataset, poisoning_label):
    return testing(model, dataset, poisoning_label=poisoning_label)
