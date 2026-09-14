import numpy as np


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    
    probabilities = counts / len(y)
    
    return 1 - np.sum(probabilities ** 2)



def split_data(X, y, feature, threshold):
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    X_left = X[left_mask]
    y_left = y[left_mask]

    X_right = X[right_mask]
    y_right = y[right_mask]

    return X_left, y_left, X_right, y_right


def split_gini(y_left, y_right):
    total = len(y_left) + len(y_right)

    left_weight = len(y_left) / total
    right_weight = len(y_right) / total

    impurity = (
        left_weight * gini(y_left)
        + right_weight * gini(y_right)
    )

    return impurity
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def find_best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    n_features = X.shape[1]

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])

        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_data(
                X, y, feature, threshold
            )

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            current_gini = split_gini(y_left, y_right)

            if current_gini < best_gini:
                best_gini = current_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gini

def build_tree(X, y, depth=0, max_depth=3):

    if len(np.unique(y)) == 1:
        return Node(value=np.bincount(y).argmax())

    if depth >= max_depth:
        return Node(value=np.bincount(y).argmax())

    feature, threshold, impurity = find_best_split(X, y)

    if feature is None:
        return Node(value=np.bincount(y).argmax())

    X_left, y_left, X_right, y_right = split_data(
        X, y, feature, threshold
    )

    left_child = build_tree(X_left, y_left, depth + 1, max_depth)
    right_child = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(
        feature=feature,
        threshold=threshold,
        left=left_child,
        right=right_child
    )

def predict_one(node, x):
    if node.value is not None:
        return node.value

    
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def predict(tree, X):
    predictions = []

    for x in X:
        predictions.append(predict_one(tree, x))

    return np.array(predictions)

X_train = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

tree = build_tree(X_train, y_train)

X_test = np.array([
    [1],
    [2.5],
    [3],
    [4],
    [5.5],
    [6]
])

predictions = predict(tree, X_test)

print("Predictions:", predictions)
