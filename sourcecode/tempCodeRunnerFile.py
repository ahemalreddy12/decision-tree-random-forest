import numpy as np


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    
    probabilities = counts / len(y)
    
    return 1 - np.sum(probabilities ** 2)


y = np.array([0, 0, 0, 1, 1])

print(gini(y))

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
y_left = np.array([0, 1, 0])
y_right = np.array([1, 1, 0])

print("Split Gini:", split_gini(y_left, y_right))