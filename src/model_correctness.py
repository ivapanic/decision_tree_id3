import collections
import itertools


def calculate_accuracy(predicted_values, true_values):
    n_total = len(true_values)
    n_correct = 0

    for predicted, true in zip(predicted_values, true_values):
        if predicted == true:
            n_correct += 1

    return n_correct / n_total


def calculate_confusion_matrix(predicted_values, true_values):
    confusion_matrix = {}
    combinations = set(itertools.product(set(true_values), set(true_values)))
    for combination in combinations:
        confusion_matrix[combination] = 0

    for predicted, true in zip(predicted_values, true_values):
        confusion_matrix[predicted, true] += 1
    confusion_matrix = collections.OrderedDict(sorted(confusion_matrix.items()))
    return confusion_matrix

