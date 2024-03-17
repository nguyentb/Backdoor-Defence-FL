import copy
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
from client import Client, inject_trigger, initialise_trigger_arr, get_device
from preprocess_dataset import train_dataset, test_dataset
import defense

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


def detect_attackers(client_tags, clients, adversaries):
    for adversary in adversaries:
        clients.append(adversary)

    for client in clients:
        if client not in client_tags.keys():
            client_tags[client] = 1
        else:
            client_tags[client] += 1

    print(client_tags)
    print()


def remove_from_training(tagged_client, benign_clients, adversaries, client_tags, results, curr_round):
    print("removing client", tagged_client, "from training, as they were tagged over the threshold.")
    results["remove"].append(curr_round)

    if tagged_client in benign_clients:
        benign_clients.remove(tagged_client)
        client_tags.pop(tagged_client)

    elif tagged_client in adversaries:
        adversaries.remove(tagged_client)
        client_tags.pop(tagged_client)

    else:
        print("Error: client was not found in list")

    return benign_clients, adversaries


def remove_attackers(client_tags, benign_clients, adversaries, results, curr_round):
    print()
    highest_clients = []
    max = list(client_tags.values())[0]
    prev_tags = copy.copy(client_tags)

    for tagged_client in prev_tags.keys():
        tag_num = prev_tags[tagged_client]
        print(str(tagged_client), "has", str(tag_num), "votes.")

        if tag_num == max:
            highest_clients.append(tagged_client)
        elif tag_num > max:
            highest_clients = [tagged_client]

        # remove all clients that have appeared more than threshold
        if tag_num > 5:
            benign_clients, adversaries = remove_from_training(tagged_client, benign_clients, adversaries, client_tags,
                                                               results, curr_round)

    if len(highest_clients) == 1:
        benign_clients, adversaries = remove_from_training(highest_clients[0], benign_clients, adversaries, client_tags,
                                                           results, curr_round)

    print()
    return benign_clients, adversaries


def server_train(attack, global_net, config, client_idcs):
    adversary_num = 0
    if attack:
        adversary_num = 1

    if config["dataset"] == "blood cell":
        print("Setting params for blood cell")
        benign_learning_rate = 0.01
        benign_decay = 0.005

        attacker_learning_rate = 0.01
        attacker_decay = 0.0005
        scheduler = False

    else:
        if config["dataset"] != "CIFAR10":
            print("WARNING!")
            print("Dataset specified not implemented. Using CIFAR10 instead.")

        benign_learning_rate = 0
        benign_decay = 0
        attacker_learning_rate = 0
        attacker_decay = 0

    results = {"train_loss_after": [], "train_loss_before": [],
               "test_loss_before": [], "test_loss_after": [],
               "test_accuracy_after": [], "test_accuracy_before": [],
               "train_accuracy_after": [], "train_accuracy_before": [],

               "backdoor_test_loss_after": [], "backdoor_test_loss_before": [],
               "backdoor_test_accuracy_before": [],
               "backdoor_test_accuracy_after": [],
               "remove": [],
               "unsure": [],
               "correct": [],
               "incorrect": [],
               "avg": [],
               "conv_avg": []
               }

    if config["load_results"]:
        results = json.load(open("logs/" + config["preload_model_name"] + ".txt"))

    best_accuracy = 0
    backdoor_t_accuracy = 0
    device = get_device()
    # defense.clean_model(global_net, device, True)
    client_tags = {}

    benign_clients = list(range(config["total_clients"] - adversary_num))
    adversaries = list(range(len(benign_clients), config["total_clients"]))

    round_adversary_num = adversary_num

    print("Benign clients: ", benign_clients)
    print("Adversaries: ", adversaries)

    for curr_round in range(config["start_round"], config["rounds"] + 1):
        attack_condition = (curr_round >= config["poisoning_epoch"]) and (curr_round % 2 == 0) and attack
        # attack_condition = True
        if attack_condition:
            print("Changing params to adapt to attack")
            benign_learning_rate = 0.05
            benign_decay = 0.005
            scheduler = False

        m = max(config["total_clients"] * config["client_num_proportion"], 1)
        print("Choosing", m, "clients.")
        print('Start Round {} ...'.format(curr_round))
        local_weights, local_loss, local_acc = [], [], []

        weight_accumulator = {}
        for name, params in global_net.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params).float()

        for adversary in adversaries:
            # force an attack at round "poisoning_epoch"
            if attack_condition:
                m = m - 1
                print("carrying out attack")
                client_update(adversary, client_idcs, config, curr_round, global_net, local_acc, local_loss,
                              local_weights, weight_accumulator, attacker_decay,
                              attacker_learning_rate, config["attacker_epochs"], False, round_adversary_num, scheduler)

        clients = list(np.random.choice(benign_clients, int(m), replace=False))
        for client in clients:

            client_update(client, client_idcs, config, curr_round, global_net, local_acc, local_loss,
                          local_weights, weight_accumulator, benign_decay, benign_learning_rate,
                          config["benign_epochs"], True, round_adversary_num, scheduler)

        model_aggregate(weight_accumulator=weight_accumulator, global_model=global_net, conf=config)

        if config["carry_defence"]:
            adversaries, benign_clients = apply_defence(adversaries, attack, attack_condition, backdoor_t_accuracy,
                                                        benign_clients, best_accuracy, client_tags, clients, config,
                                                        curr_round, device, global_net, local_acc, local_loss, results)

        test_aggregated_model(attack, backdoor_t_accuracy, best_accuracy, config, global_net, results, local_acc,
                              local_loss, "after")

        save_model(config, curr_round, global_net)


def apply_defence(adversaries, attack, attack_condition, backdoor_t_accuracy, benign_clients, best_accuracy,
                  client_tags, clients, config, curr_round, device, global_net, local_acc, local_loss, results):
    state_before = copy.deepcopy(global_net.state_dict())
    test_aggregated_model(attack, backdoor_t_accuracy, best_accuracy, config, global_net, results, local_acc,
                          local_loss, "before")

    poisoned = defense.clean_model(global_net, device, test_dataset, config["dataset"], results)

    if poisoned == 0:
        if not attack_condition or len(adversaries) == 0:
            print("Server CORRECTLY believes model was clean.")
            results["correct"].append(curr_round)

        else:
            print("Server INCORRECTLY believes model was clean.")
            results["incorrect"].append(curr_round)

        global_net.load_state_dict(state_before)

    elif poisoned == 1:
        print()
        print("Server received notice that model was poisoned")

        if attack_condition and len(adversaries) > 0:
            print("Attack was correctly detected.")
            results["correct"].append(curr_round)

        else:
            print("Attack was incorrectly detected.")
            results["incorrect"].append(curr_round)

        detect_attackers(client_tags, clients, adversaries)
        benign_clients, adversaries = remove_attackers(client_tags, benign_clients, adversaries, results,
                                                       curr_round)

        if not adversaries:
            print("All adversaries have been removed.")
            adversaries = []

    elif not attack_condition or len(adversaries) == 0:
        results["unsure"].append(curr_round)
        print("Model was not poisoned and model was not sure.")

    elif attack_condition:
        results["unsure"].append(curr_round)
        print("Model was poisoned and model was not sure.")

    return adversaries, benign_clients


def save_model(config, curr_round, global_net):
    if curr_round < config["poisoning_epoch"]:
        torch.save(global_net.state_dict(), "pretrained_models/" + config["log_file"] + "_no_attack.pt")
        # open("results_from_beginning.txt", 'w').write(json.dumps(results))

    else:
        torch.save(global_net.state_dict(), "pretrained_models/" + config["log_file"] + ".pt")


def test_aggregated_model(attack, backdoor_t_accuracy, best_accuracy, config, global_net, results, local_acc,
                          local_loss, time):
    train_acc = sum(local_acc) / len(local_acc)
    train_loss = sum(local_loss) / len(local_loss)
    results["train_accuracy_" + time].append(train_acc)
    results["train_loss_" + time].append(train_loss.item())

    t_accuracy = defense.testing(global_net, test_dataset)
    results["test_accuracy_" + time].append(t_accuracy)
    # results["test_loss_" + time].append(t_loss)

    print("Finished benign test")

    if attack:
        backdoor_t_accuracy = defense.poisoned_testing(global_net, test_dataset)
        results["backdoor_test_accuracy_" + time].append(backdoor_t_accuracy)
        # results["backdoor_test_loss_" + time].append(backdoor_t_loss)

    if best_accuracy < t_accuracy:
        best_accuracy = t_accuracy

    print("TRAIN ACCURACY", train_acc)
    print()
    print("BACKDOOR:", backdoor_t_accuracy)
    print("MAIN ACCURACY:", t_accuracy)
    print()

    open("logs/" + config["log_file"] + ".txt", 'w').write(json.dumps(results))


def client_update(client, client_idcs, config, curr_round, global_net, local_acc, local_loss, local_weights,
                  weight_accumulator, decay, learning_rate, client_epochs, benign, round_adversary_num, scheduler):
    adversary_update = Client(dataset=train_dataset, batch_size=config["batch_size"],
                              client_id=client_idcs[client],
                              benign=benign, epochs=client_epochs, config=config)

    for i in range(len(config["lr_decrease_epochs"])):
        if curr_round > config["lr_decrease_epochs"][i]:
            learning_rate *= 0.5
        else:
            continue
    weights, loss, train_acc = adversary_update.train(model=copy.deepcopy(global_net),
                                                      lr=learning_rate,
                                                      decay=decay, round_adversary_num=round_adversary_num, set_scheduler=scheduler)
    local_weights.append(copy.deepcopy(weights))
    local_loss.append(loss)
    local_acc.append(train_acc)

    for name, params in global_net.state_dict().items():
        weight_accumulator[name].add_(weights[name])
