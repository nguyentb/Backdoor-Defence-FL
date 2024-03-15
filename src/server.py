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


def remove_from_training(tagged_client, benign_clients, adversaries, client_tags):
    print("removing client", tagged_client, "from training, as they were tagged over the threshold.")

    if tagged_client in benign_clients:
        benign_clients.remove(tagged_client)
        client_tags.pop(tagged_client)

    elif tagged_client in adversaries:
        adversaries.remove(tagged_client)
        client_tags.pop(tagged_client)

    else:
        print("Error: client was not found in list")

    return  benign_clients, adversaries


def remove_attackers(client_tags, benign_clients, adversaries):
    print()
    highest_clients = []
    max = list(client_tags.values())[0]
    prev_tags = copy.copy(client_tags)

    for tagged_client in prev_tags.keys():
        tag_num = prev_tags[tagged_client]
        print(str(tagged_client), "has", str(tag_num), "votes.")

        if tag_num ==  max:
            highest_clients.append(tagged_client)
        elif tag_num > max:
            highest_clients = [tagged_client]

        # remove all clients that have appeared more than threshold
        if tag_num > 5:
             benign_clients, adversaries = remove_from_training(tagged_client, benign_clients, adversaries, client_tags)

    if len(highest_clients) == 1:
        benign_clients, adversaries = remove_from_training(highest_clients[0], benign_clients, adversaries,client_tags)

    print()
    return benign_clients, adversaries

def server_train(attack, global_net, config, client_idcs):
    adversary_num = 0
    if attack:
        adversary_num = 1

    results = {"train_loss": [],
               "test_loss": [],
               "test_accuracy": [],
               "train_accuracy": [],
               "backdoor_test_loss": [],
               "backdoor_test_accuracy": []}

    if config["preload_model"]:
        results = json.load(open("logs/"+config["preload_model_name"]+".txt"))

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

    for curr_round in range(1, config["rounds"] + 1):
        # attack_condition = config["poisoning_epoch"]
        attack_condition = True

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
                              local_weights, weight_accumulator, config["attacker_decay"],
                              config["attacker_learning_rate"], config["attacker_epochs"], False, round_adversary_num)

        clients = list(np.random.choice(benign_clients, int(m), replace=False))
        for client in clients:
            # learning_rate = config["benign_learning_rate"]
            # for i in range(len(config["lr_decrease_epochs"])):
            #     if curr_round > config["lr_decrease_epochs"][i]:
            #         learning_rate *= 0.5
            #     else:
            #         continue

            client_update(client, client_idcs, config, curr_round, global_net, local_acc, local_loss,
                          local_weights, weight_accumulator, config["benign_decay"], config["benign_learning_rate"],
                          config["benign_epochs"], True)

        model_aggregate(weight_accumulator=weight_accumulator, global_model=global_net, conf=config)

        state_before = copy.deepcopy(global_net.state_dict())
        poisoned = defense.clean_model(global_net, device)

        if poisoned == 0:
            if not attack_condition or len(adversaries) == 0:
                print("Server CORRECTLY believes model was clean.")

            else:
                print("Server INCORRECTLY believes model was clean.")

            global_net.load_state_dict(state_before)

        elif poisoned == 1:
            print()
            print("Server received notice that model was poisoned")

            if attack_condition and len(adversaries) > 0:
                print("Attack was correctly detected.")

            else:
                print("Attack was incorrectly detected.")

            detect_attackers(client_tags, clients, adversaries)
            benign_clients, adversaries = remove_attackers(client_tags, benign_clients, adversaries)

            if not adversaries:
                print("All adversaries have been removed.")
                adversaries = []

        elif not attack_condition or len(adversaries) == 0:
            print("Model was not poisoned and model was not sure.")

        elif attack_condition:
            print("Model was poisoned and model was not sure.")

        test_aggregated_model(attack, backdoor_t_accuracy, best_accuracy, config, global_net, results, local_acc,
                              local_loss)

        save_model(config, curr_round, global_net)


def save_model(config, curr_round, global_net):
    if curr_round < config["poisoning_epoch"]:
        torch.save(global_net.state_dict(), "pretrained_models/"+config["log_file"]+"_no_attack.pt")
        # open("results_from_beginning.txt", 'w').write(json.dumps(results))

    else:
        torch.save(global_net.state_dict(), "pretrained_models/"+config["log_file"]+".pt")


def test_aggregated_model(attack, backdoor_t_accuracy, best_accuracy, config, global_net, results, local_acc,
                          local_loss):
    train_acc = sum(local_acc) / len(local_acc)
    train_loss = sum(local_loss) / len(local_loss)
    results["train_accuracy"].append(train_acc)
    results["train_loss"].append(train_loss.item())

    t_accuracy, t_loss = testing(global_net, test_dataset)
    results["test_accuracy"].append(t_accuracy)
    results["test_loss"].append(t_loss)

    print("Finished benign test")

    if attack:
        backdoor_t_accuracy, backdoor_t_loss = poisoned_testing(global_net, test_dataset, config["poisoning_label"])
        results["backdoor_test_accuracy"].append(backdoor_t_accuracy)
        results["backdoor_test_loss"].append(backdoor_t_loss)

    if best_accuracy < t_accuracy:
        best_accuracy = t_accuracy

    print("TRAIN ACCURACY", train_acc)
    print()
    print("BACKDOOR:", backdoor_t_accuracy)
    print("MAIN ACCURACY:", t_accuracy)
    print()

    open("logs/"+config["log_file"]+".txt", 'w').write(json.dumps(results))


def client_update(client, client_idcs, config, curr_round, global_net, local_acc, local_loss, local_weights,
                  weight_accumulator, decay, learning_rate, client_epochs, benign, round_adversary_num):
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
                                                      decay=decay, round_adversary_num=round_adversary_num)
    local_weights.append(copy.deepcopy(weights))
    local_loss.append(loss)
    local_acc.append(train_acc)

    for name, params in global_net.state_dict().items():
        weight_accumulator[name].add_(weights[name])


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
