import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "sourcecode")
    )
)

import numpy as np
from decisiontree import (
    gini,
    split_data,
    split_gini,
    find_best_split,
    build_tree,
    predict,
    accuracy,
    bootstrap_sample,
    RandomForest
)


# Test Gini impurity
y = np.array([0, 0, 0, 1, 1])
assert abs(gini(y) - 0.48) < 1e-6


# Test dataset splitting
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([0, 0, 0, 1, 1, 1])

X_left, y_left, X_right, y_right = split_data(
    X, y, 0, 3
)

assert len(X_left) == 3
assert len(X_right) == 3


# Test split impurity
assert split_gini(
    np.array([0, 0]),
    np.array([1, 1])
) == 0.0


# Test best split
feature, threshold, impurity = find_best_split(X, y)

assert feature == 0
assert threshold == 3
assert impurity == 0.0


# Test Decision Tree
tree = build_tree(X, y, max_depth=3)

predictions = predict(tree, X)

assert accuracy(y, predictions) == 1.0


# Test bootstrap sampling
X_sample, y_sample = bootstrap_sample(X, y)

assert len(X_sample) == len(X)
assert len(y_sample) == len(y)


# Test Random Forest
forest = RandomForest(n_trees=5, max_depth=3)

forest.fit(X, y)

forest_predictions = forest.predict(X)

assert len(forest_predictions) == len(y)

print("All tests pased")