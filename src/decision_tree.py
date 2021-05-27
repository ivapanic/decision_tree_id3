import collections
import math
import operator
from copy import deepcopy


class ID3:
    feature_values = {}
    feature_values_test = {}
    outcome_values = set()
    outcome = ""
    decision_tree = {}
    predictions = []

    def __init__(self, data, data_test, max_depth):
        self.data = data
        self.__prepare_dataset_for_calculation()
        self.data_test = data_test
        self.__prepare_test_dataset_for_calculation()
        self.max_depth = max_depth

    def __prepare_dataset_for_calculation(self):
        features = list(self.data[0].keys())
        self.outcome = features.pop()

        for feature in features:
            for line in self.data:
                if feature not in self.feature_values:
                    values = set()
                    values.add(line[feature])
                    self.feature_values[feature] = values
                else:
                    self.feature_values[feature].add(line[feature])
        for line in self.data:
            if self.outcome in line:
                self.outcome_values.add(line[self.outcome])

    def __prepare_test_dataset_for_calculation(self):
        features = list(self.data_test[0].keys())
        features.pop()

        for feature in features:
            for line in self.data_test:
                if feature not in self.feature_values_test:
                    values = set()
                    values.add(line[feature])
                    self.feature_values_test[feature] = values
                else:
                    self.feature_values_test[feature].add(line[feature])

    def __get_entropy(self, data):
        entropy = 0
        n_instances_outcomes = {}
        sum_outcomes = len(data)
        for line in data:
            if not line[self.outcome] in n_instances_outcomes:
                n_instances_outcomes[line[self.outcome]] = 1
            else:
                n_instances_outcomes[line[self.outcome]] += 1
        for number_instances in n_instances_outcomes.values():
            entropy -= (number_instances / sum_outcomes) * math.log2(number_instances / sum_outcomes)
        return entropy

    def __calculate_num_instances_outcomes(self, data, current_feature):
        n_instances_outcomes = {}
        for line in data:
            if not line[current_feature] in n_instances_outcomes:
                values = {}
                n_instances_outcomes[line[current_feature]] = values
            if not line[self.outcome] in n_instances_outcomes[line[current_feature]]:
                n_instances_outcomes[line[current_feature]][line[self.outcome]] = 1
            else:
                n_instances_outcomes[line[current_feature]][line[self.outcome]] += 1
        return n_instances_outcomes

    def __get_information_gain(self, current_feature, data):
        information_gain = self.__get_entropy(data)
        sum_instances_all = len(data)
        n_instances_outcomes = self.__calculate_num_instances_outcomes(data, current_feature)

        for feature_value, n_feature_outcomes in n_instances_outcomes.items():
            sum_outcomes = 0
            for line in data:
                if line[current_feature] == feature_value:
                    sum_outcomes += 1
            for number_instances in n_feature_outcomes.values():
                entropy = - (number_instances / sum_outcomes) * math.log2(number_instances / sum_outcomes)
                information_gain -= (sum_outcomes / sum_instances_all) * entropy
        return information_gain

    def __get_max_information_gain_feature(self, feature_values, data):
        features_gain = {}
        if feature_values:
            for key in feature_values.keys():
                features_gain[key] = self.__get_information_gain(key, data)
                print("IG(" + key + ") = %.4f" % features_gain[key])
            features_gain = collections.OrderedDict(sorted(features_gain.items()))
            return max(features_gain.items(), key=operator.itemgetter(1))[0]

    def __find_most_probable_outcome(self, data):
        classification = {}
        if data:
            for line in data:
                if not line[self.outcome] in classification:
                    classification[line[self.outcome]] = 1
                else:
                    classification[line[self.outcome]] += 1
            return max(classification.items(), key=operator.itemgetter(1))[0]

    def __get_updated_values(self, feature_values, used_features):
        feature_values_next = feature_values
        for used_feature in used_features:
            if used_feature in feature_values_next:
                del feature_values_next[used_feature]
        return feature_values

    def __id3(self, data, parent_data, used_features, feature_names, feature_values, depth):
        tree = {}
        for line in data:
            self.outcome_values.add(line[self.outcome])
        if depth != self.max_depth:
            if len(data) == 0:
                return self.__find_most_probable_outcome(parent_data)
            if len(self.outcome_values) == 1:
                return self.__find_most_probable_outcome(data)
            else:
                if used_features:
                    feature_values_next = self.__get_updated_values(feature_values, used_features)
                    node = self.__get_max_information_gain_feature(feature_values_next, data)
                    if not node:
                        data.clear()
                        return self.__id3(data, parent_data, used_features, feature_names, feature_values, depth + 1)
                else:
                    node = self.__get_max_information_gain_feature(feature_values, data)
            values = feature_values[node]
            tree[node] = {}
            for value in values:
                new_data = []
                tree[node][value] = {}
                for line in data:
                    if line[node] == value:
                        new_data.append(line)
                used_features.add(node)
                self.outcome_values.clear()
                subtree = self.__id3(new_data, data, used_features, feature_names, feature_values, depth + 1)
                tree[node][value] = subtree
            return tree
        else:
            if data:
                return self.__find_most_probable_outcome(data)
            else:
                return self.__find_most_probable_outcome(parent_data)

    def fit(self):
        feature_names = set()
        used_features = set()
        feature_values = deepcopy(self.feature_values)
        for feature_name in feature_values.keys():
            feature_names.add(feature_name)
        self.decision_tree = self.__id3(self.data, self.data, used_features, feature_names, feature_values, 0)
        return self.decision_tree

    def __find_feature_in_tree(self, item, decision_tree):
        for key, value in decision_tree.items():
            if key == item:
                return True
            if isinstance(value, dict):
                value_items = value.items()
                for value_key, value_val in value_items:
                    if isinstance(value_val, str):
                        break
                    if value_key == item:
                        return True
                    self.__find_feature_in_tree(item, value_val)

    def __search_tree(self, line, feature_values_test, data_test, decision_tree):
        outcome = self.__find_most_probable_outcome(data_test)
        for feature in line:
            if not self.__find_feature_in_tree(feature, decision_tree):
                continue
            if feature not in self.feature_values:
                return outcome
            else:
                curr_feature_value = line[feature]
                subtree = decision_tree[feature][curr_feature_value]
                if isinstance(subtree, str):
                    return subtree
                else:
                    return self.__search_tree(line, feature_values_test, data_test, subtree)

    def predict(self):
        feature_values_test = self.feature_values_test
        data_test = self.data_test
        decision_tree = self.decision_tree
        for line in data_test:
            prediction = self.__search_tree(line, feature_values_test, data_test, decision_tree)
            self.predictions.append(prediction)
        return self.predictions

    def get_true_values_test(self):
        true_values = []
        for line in self.data_test:
            true_values.append(line[self.outcome])
        return true_values
