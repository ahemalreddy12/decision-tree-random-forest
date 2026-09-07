import numpy as np


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    
    probabilities = counts / len(y)
    
    return 1 - np.sum(probabilities ** 2)


y = np.array([0, 0, 0, 1, 1])

print(gini(y))