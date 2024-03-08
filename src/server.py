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
    backdoor_t_accuracy = 0
    for curr_round in range(1, config["rounds"] + 1):
        m = config["total_clients"] * config["client_num_proportion"]
        print("Choosing", m, "clients.")
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, local_acc = [], [], []

        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params).float()

        for adversary in range(1, adversaries + 1):
            if curr_round == 1:
            # if curr_round == config["poisoning_epoch"]:
                m = m - 1
                print("carrying out attack")
                adversary_update = Client(dataset=train_dataset, batch_size=config["batch_size"],
                                          client_id=client_idcs[-adversary],
                                          benign=False, epochs=config["attacker_epochs"], config=config)

                learning_rate = config["attacker_learning_rate"]
                for i in range(len(config["lr_decrease_epochs"])):
                    if curr_round > config["lr_decrease_epochs"][i]:
                        learning_rate *= 0.5
                    else:
                        continue
                weights, loss, train_acc = adversary_update.train(model=copy.deepcopy(global_net),
                                                                  lr=learning_rate,
                                                                  decay=config["attacker_decay"])

                local_weights.append(copy.deepcopy(weights))
                local_loss.append(loss)
                local_acc.append(train_acc)

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
            weights, loss, train_acc = local_update.train(model=copy.deepcopy(global_net),
                                                          lr=learning_rate, decay=config["benign_decay"])

            local_weights.append(copy.deepcopy(weights))
            local_loss.append(loss)
            local_acc.append(train_acc)

            for name, params in global_net.state_dict().items():
                weight_accumulator[name].add_(weights[name])

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
        if curr_round < config["poisoning_epoch"]:
            torch.save(global_net.state_dict(), "pretrained_models/from_beginning_before_attack.pt")
            # open("results_from_beginning.txt", 'w').write(json.dumps(results))

        else:
            torch.save(global_net.state_dict(), "pretrained_models/from_beginning_after_attack.pt")

        open("logs/results_from_beginning_lr1.txt", 'w').write(json.dumps(results))

        print("TRAIN ACCURACY", train_acc.item())
        print()
        print("BACKDOOR:", backdoor_t_accuracy)
        print("MAIN ACCURACY:", t_accuracy)
        print()


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
            labels, imgs = inject_trigger(imgs, labels, poisoning_label, pos, len(imgs))

        if torch.cuda.is_available():
            imgs, labels = imgs.cuda(), labels.cuda()

        output = model(imgs)

        loss = loss_function(output, labels)
        loss_sum += loss.item()

        prediction = torch.max(output, 1)

        correct_num += (labels == prediction[1]).sum().item()
        sample_num += labels.shape[0]

    accuracy = 100 * correct_num / sample_num

    return accuracy, loss_sum


def poisoned_testing(model, dataset, poisoning_label):
    return testing(model, dataset, poisoning_label=poisoning_label)
