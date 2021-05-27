def print_tree(tree):
    print("[BRANCHES]:")
    print_tree_recursively(tree, 1, "")


def print_tree_recursively(tree, depth, output):
    for key, value in tree.items():
        if isinstance(value, dict):
            for value_key in value.keys():
                value_val = value[value_key]
                if isinstance(value_val, str):
                    output_add = output + str(depth) + ":" + key + "=" + value_key + " " + value_val
                    print(output_add)
                else:
                    new_output = output
                    output += str(depth) + ":" + key + "=" + value_key + " "
                    print_tree_recursively(value_val, depth + 1, output)
                    output = new_output


def print_predictions(predictions):
    print("[PREDICTIONS]: " + " ".join(predictions))


def print_accuracy(accuracy):
    print("[ACCURACY]: %.5f" % accuracy)


def print_confusion_matrix(confusion_matrix, n_classes):
    print("[CONFUSION_MATRIX]:")
    output = []
    for value in confusion_matrix.values():
        output.append(value)

    for i in range(n_classes):
        s = ""
        for j in range(n_classes):
            s += str(output[j * n_classes + i]) + " "
        print(s)







