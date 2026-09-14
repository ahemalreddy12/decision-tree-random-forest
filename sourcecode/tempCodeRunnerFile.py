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

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([0, 0, 0, 1, 1, 1])

X_left, y_left, X_right, y_right = split_data(X, y, 0, 3.5)

print("Left:")
print(X_left)
print(y_left)

print("Right:")
print(X_right)
print(y_right)