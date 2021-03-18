import argparse
import time
from csv import reader
from math import floor
from random import randrange, shuffle


def main(filename: str, n_trees: int, with_root: bool):
    return mpi_root(filename, n_trees, with_root)


##################################################################################################################################

# All functions


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += proportion * (1.0 - proportion)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            # for i in range(len(dataset)):
            #     row = dataset[randrange(len(dataset))]
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del node["groups"]
    # check for a no split
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split(node["left"], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split(node["right"], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, test):
    predictionS = []
    for row in test:
        prediction = [predict(tree, row) for tree in trees]
        predictionS.append(max(set(prediction), key=prediction.count))
    return predictionS


# Bootstrap Aggregation Algorithm
def bagging(train, max_depth, min_size, sample_size, n_trees):
    trees = list()
    start_time = time.time()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    log(f"finished {i + 1} trees in {time.time() - start_time:.1f} sec")
    return trees


def formate_dataset(dataset, percent_test):
    shuffle(dataset)
    test_set = [dataset.pop() for _ in range(floor(len(dataset) * percent_test))]
    return (dataset, test_set)


def evaluate_algorithm(train, test, trees):
    predicted = bagging_predict(trees, test)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predicted)
    return (accuracy, predicted)


#######################################################################################################################################


def mpi_root(filename: str, n_trees: int, with_root: bool = True):
    # Ce que fait le root
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0]) - 1)
    train, test = formate_dataset(list(dataset), 0.2)

    data = [train, 6, 2, 0.5, n_trees]

    start_time = time.time()

    trees = bagging(*data)

    accuracy, _ = evaluate_algorithm(train, test, trees)

    duration = time.time() - start_time

    log(f"Accuracy: {accuracy:.2f}")

    print(f"\nTime: {duration:.2f}")


def log(msg: str) -> None:
    print(f"[info] - {msg}")



if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', help='csv file', type=str)
    parser.add_argument('--trees', '-t', help='number of trees', default=5, type=int)
    parser.add_argument('--root', '-r', help='Also build trees on the master node', action="store_true")
    args = parser.parse_args()

    sys.exit(main(args.file, args.trees, args.root))
