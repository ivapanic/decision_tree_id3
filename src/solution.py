import dataset_parser
from decision_tree import ID3
from output import *
from model_correctness import *
import sys

def main():
    #path_dataset_train = "./datasets/heldout_unseen_featurevalue_train.csv"
    #path_dataset_test = "./datasets/heldout_unseen_featurevalue_test.csv"
    #max_depth = 1

    path_dataset_train = sys.argv[1]
    path_dataset_test = sys.argv[2]
    if len(sys.argv) == 4:
        max_depth = int(sys.argv[3])
    else:
        max_depth = -1

    data_train = dataset_parser.parse(path_dataset_train)
    data_test = dataset_parser.parse(path_dataset_test)

    model = ID3(data_train, data_test, max_depth)
    decision_tree = model.fit()

    print_tree(decision_tree)

    predictions = model.predict()
    print_predictions(predictions)

    true_values = model.get_true_values_test()

    accuracy = calculate_accuracy(predictions, true_values)
    print_accuracy(accuracy)

    confusion_matrix = calculate_confusion_matrix(predictions, true_values)
    print_confusion_matrix(confusion_matrix, len(set(true_values)))



if __name__ == '__main__':
    main()
