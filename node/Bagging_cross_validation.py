from csv import reader
from math import floor
from random import randrange, shuffle
from typing import Union

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def main():
    if mpi_size == 1:
        print("WARNING: There is only one node")
        return 1

    if mpi_size >= 1000 and mpi_rank == 0:
        print("WARNING: Your world size is {}, it's so big".format(mpi_size))

    if mpi_rank == 0:
        return mpi_root()
    else:
        return mpi_nonroot()

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
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    log(f"finished {i} trees")
    return trees


def formate_dataset(dataset, percent_test):
    shuffle(dataset)
    test_set = [dataset.pop() for _ in range(floor(len(dataset) * percent_test))]
    return (dataset, test_set)


def evaluate_algorithm(train, test, trees):
    predicted = bagging_predict(trees, test)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy


#######################################################################################################################################


def mpi_root():
    # Ce que fait le root
    filename = "node/sonar.all-data"
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, len(dataset[0]) - 1)


    n_folds = mpi_size - 1
    n_trees = 5
    max_depth = 6
    min_size = 2
    sample_size = 0.5 

    folds = cross_validation_split(dataset, n_folds)

    for i in range (len(folds)):
        data = [folds, max_depth, min_size, sample_size, n_trees] 
        mpi_comm.send(data, dest=i+1, tag=9)
    log("dataset has been sent to all the nodes")

    scores = list()
    for i in range (len(folds)):
        scores.extend(mpi_comm.recv(source=i+1, tag=11))
    log("accuracy are back")
    log('Trees: %d' % n_trees)
    log('Scores: %s' % scores)
    log('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

def log(msg: str) -> None:
    print(f"[{mpi_comm.rank}] - {msg}")


def mpi_nonroot():
    # Ce que font les autres nodes
    data = mpi_comm.recv(source=0, tag=9)
    log("dataset received")
    train_set = list(data[1])
    train_set.remove(data[1][mpi_rank])
    train_set = sum(train_set, [])
    data[1][mpi_rank] = train_set
    test_set = list()
    for row in data[1][mpi_rank]:
        row_copy = list(row)
        test_set.append(row_copy)
        row_copy[-1] = None
    trees = bagging(*data)
    accuracy = evaluate_algorithm(train_set, test_set, trees)
    mpi_comm.send(trees, dest=0, tag=11)
    log("sending the accuracy back")


if __name__ == "__main__":
    import sys

    sys.exit(main())
