import numpy as np

from node import Node


def get_thresholds(data):
    """
    :param data: 2d numeric array
    :return: A 2d array with with columns containing unique values from corresponding columns from `data`
    """
    return [sorted(set(data[:, c])) for c in range(0, data.shape[1])]


def get_predictions(data, labels):
    """
    Calculates the "Likelihood" for each label to occur
    :param data: data for which predictions are calculated
    :param labels: set of labels
    :return: dict {label_id: "Likelihood"}
    """
    return {int(l): list(data[:, -1]).count(l) / len(data[:, -1]) for l in labels}


def get_gini(predictions):
    """
    Calculates gini index based on predictions
    :param predictions:
    :return: gini index
    """
    from functools import reduce
    return reduce(lambda acc, val: acc - val ** 2, predictions.values(), 1)


def split(data, thresholds):
    """
    Tries to split data at every point in `thresholds`. Returns split with the lowest gini index
    :param data: data to split
    :param thresholds: possible points to split
    :return: split data, value and its column on which data was split
    """
    left_data = None
    right_data = None
    min_cost = 1
    column = -1
    pivot = None
    for i, feature in enumerate(thresholds[:-1]):
        for val in feature:
            data_l = np.empty((0, data.shape[1]))
            data_r = np.empty((0, data.shape[1]))
            for row in data:
                if row[i] <= val:
                    data_l = np.vstack((data_l, row))
                else:
                    data_r = np.vstack((data_r, row))

            gini_left = get_gini(get_predictions(data_l, thresholds[-1])) if data_l.shape[0] > 0 else 0
            gini_right = get_gini(get_predictions(data_r, thresholds[-1])) if data_r.shape[0] > 0 else 0
            cost = gini_left * (data_l.shape[0] / data.shape[0]) + gini_right * (data_r.shape[0] / data.shape[0])
            if min_cost is None or cost < min_cost:
                left_data = data_l
                right_data = data_r
                column = i
                pivot = val
                min_cost = cost

    return left_data, right_data, column, pivot


def fit_tree(data, depth):
    """
    Recursively creates decision tree
    :param data: data at current node
    :param depth: current depth
    :return: root of (sub)tree
    """
    root = Node()
    thresholds = get_thresholds(data)
    root.prediction = get_predictions(data, thresholds[-1])
    root.misclassification_rate = get_gini(root.prediction)
    data_left, data_right, root.column, root.pivot = split(data, thresholds)
    if depth >= 2 or len(root.prediction) == 1:
        return root
    root.left = fit_tree(data_left, depth + 1)
    root.right = fit_tree(data_right, depth + 1)
    return root


def print_tree(root, indentation):
    """
    :param root: Root of the tree
    :param indentation: Each level in tree adds indentation string for clearer output
    """
    if root is not None:
        prediction = {el: round(root.prediction[el], 3) for el in root.prediction}
        print_tree(root.right, "    {}->".format(indentation))
        print("{} \\textcolor{{blue}}{{ Distribution: ({}), gini index: {} }} {}"
              .format(indentation,
                      prediction,
                      round(root.misclassification_rate, 3),
                      "column: {}, pivot: {}".format(root.column, root.pivot) if not root.is_leaf() else "LEAF"))
        print_tree(root.left, "    {}->".format(indentation))


def get_class(prediction):
    """

    :param prediction: predictions for each label
    :return: the most likely label and its prediction
    """
    label = max(prediction, key=prediction.get)
    return label, round(prediction[label],3)


def classify(root, vector):
    """
    Recursively travers a tree to find label of `vector`
    :param root: root of (sub)tree
    :param vector: vector to classify
    :return: label
    """
    if root.left is None and root.right is None:
        return get_class(root.prediction)
    pivot = vector[root.column]
    if pivot <= root.pivot:
        return classify(root.left, vector)
    else:
        return classify(root.right, vector)


data = np.loadtxt("data/01_homework_dataset.csv", delimiter=",", skiprows=1)
tree_root = fit_tree(data, 0)
print_tree(tree_root, "")

x_a = [4.1, -0.1, 2.2]
x_b = [6.1, 0.4, 1.3]
print("x_a class: {}".format(classify(tree_root, x_a)))
print("x_b class: {}".format(classify(tree_root, x_b)))
