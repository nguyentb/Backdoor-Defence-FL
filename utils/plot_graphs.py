import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import data

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 13})
plt.rcParams["figure.figsize"] = (12,5)

colours = {"test accuracy without attack": "b",
           "test accuracy": "g",
           "backdoor accuracy": "r",
           "test accuracy after": "yellowgreen",
           "backdoor accuracy after": "hotpink",
           "train accuracy": "blue"}


def show_plot():
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.subplots_adjust(right=0.7)
    plt.show()


def show_attack_rounds(ax1, results):
    """Add dashed lines that indicate when an attack happened."""
    for att in results["_with_attack"]["attack_rounds"][:-1]:
        ax1.axvline(x=int(att), color='orange', linestyle="dashed")

    ax1.axvline(x=int(results["_with_attack"]["attack_rounds"][-1]), color='orange', linestyle="dashed",
                label="Attack occurred")

    if "removed_round" in results:
        ax1.axvline(x=int(results["_with_attack"]["removed_round"][0]), color='purple', linestyle="dashed",
                    label="All attackers were removed.")


def draw_graph(ax1, results, x_axis, attack, marker="", linestyle="solid", x_label="Communication rounds after the first attack"):
    for result in results.keys():
        for metric in results[result].keys():
            values = results[result][metric]

            conditions = "backdoor" not in metric and "without" in result
            if attack:
                conditions = "train" not in metric

            if len(values) > 0 and "accuracy" in metric and conditions:
                if "backdoor" in metric:
                    label = "backdoor accuracy"
                    if "after" in metric:
                        label = "backdoor accuracy after"
                elif "train" in metric:
                    label = "train accuracy"

                elif "without" in result:
                    label = "test accuracy"

                else:
                    label = "test accuracy"
                    if "after" in metric:
                        label = "test accuracy after"

                y_axis = np.array(values)
                ax1.plot(x_axis, y_axis, colours[label], label=label.capitalize(), marker=marker, linestyle=linestyle)
        ax1.set(xlabel=x_label, ylabel='Accuracy (%)')

    ax1.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


def draw_baseline(ax1, ax2):
    # CIFAR10
    rounds = len(data.base_acc_cifar_and_every_ten["_with_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    result = data.base_acc_cifar_and_every_ten

    draw_graph(ax1, result, x_axis, False, x_label="Communication rounds")

    # BLOOD CELL
    results = data.base_acc_blood_cell
    rounds = len(results["_without_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax2, results, x_axis, False, x_label="Communication rounds")
    ax2.set_xticks(ticks=list(range(0, rounds + 2, 50)), labels=list(range(0, rounds + 2, 50)))
    show_plot()


def draw_one_shot(ax1, ax2):
    # CIFAR10
    results = data.one_shot_cifar
    rounds = len(results["_with_attack"]["test_accuracy_before"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True)
    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=list(range(1, rounds + 2, 50)), labels=[-1] + list(range(50, rounds + 2, 50)))

    # BLOOD CELL
    results = data.blood_cell_oneShot_decay
    rounds = len(results["_with_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax2, results, x_axis, True)
    show_attack_rounds(ax2, results)

    ax2.set_xticks(ticks=list(range(1, rounds + 2, 5)), labels=[-1] + list(range(5, rounds + 2, 5)))
    show_plot()


def draw_few_shot(ax1, ax2):
    # CIFAR10
    results = data.CIFAR_after_convergence_attack_every_other_no_defense
    rounds = len(results["_with_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True)
    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=[1, 2] + list(range(4, rounds + 2, 2)), labels=[-1, 0] + list(range(2, rounds, 2)))

    # BLOOD CELL
    results = data.blood_cell_one_attacker_every_other_round_after_conv
    rounds = len(results["_with_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax2, results, x_axis, True)
    show_attack_rounds(ax2, results)

    ax2.set_xticks(ticks=[1, 2] + list(range(4, rounds + 2, 2)), labels=[-1, 0] + list(range(2, rounds, 2)))

    show_plot()


def draw_defense_every_2(ax1):
    results = data.one_attacker_every_other_round_defense_after_conv
    rounds = len(results["_with_attack"]["test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True)
    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=[1, 2] + list(range(4, rounds + 2, 2)), labels=[-1, 0] + list(range(2, rounds, 2)))

    show_plot()


def draw_defense_every_10(ax1):
    results = data.one_attacker_defense_every_ten

    rounds = len(results["_with_attack"]["backdoor_test_accuracy_before"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True, "o", "")

    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=list(range(1, rounds + 2, 10)), labels=list(range(0, rounds+1, 10)))

    show_plot()


def draw_removal_one_attacker(ax1):
    results = data.one_attacker_every_round_after_conv

    rounds = len(results["_with_attack"]["backdoor_test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True)

    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=list(range(1, rounds + 2, 10)), labels=list(range(0, rounds + 1, 10)))

    show_plot()


def draw_removal_ten_attacker(ax1):
    results = data.ten_attackers_after_conv_6_threshold

    rounds = len(results["_with_attack"]["backdoor_test_accuracy"])
    x_axis = np.arange(1, rounds + 1)
    draw_graph(ax1, results, x_axis, True)

    show_attack_rounds(ax1, results)

    ax1.set_xticks(ticks=list(range(1, rounds + 2, 10)), labels=list(range(0, rounds + 1, 10)))

    show_plot()


# BASELINE ACCURACIES
_, (ax1, ax2) = plt.subplots(1, 2)
draw_baseline(ax1, ax2)

# ONE-SHOT
_, (ax1, ax2) = plt.subplots(1, 2)
draw_one_shot(ax1, ax2)

# FEW-SHOT
_, (ax1, ax2) = plt.subplots(1, 2)
draw_few_shot(ax1, ax2)

# Effect of backdoor unlearning - attacker every other round
_, ax1 = plt.subplots(1, 1)
draw_defense_every_2(ax1)

# Behaviour before and after unlearning
_, ax1 = plt.subplots(1, 1)
draw_defense_every_10(ax1)

# Removing participants - one attacker
_, ax1 = plt.subplots(1, 1)
draw_removal_one_attacker(ax1)

# Removing participants - ten attackers
_, ax1 = plt.subplots(1, 1)
draw_removal_ten_attacker(ax1)

