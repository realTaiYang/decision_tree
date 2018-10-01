import numpy as np
import pandas as pd


class Node:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.pts = self.X.shape[0]
        self.dim = self.X.shape[1]
        self._addcls()

    # Add classes
    def _addcls(self):
        self.cls = []
        for cl in self.y:
            if cl not in self.cls:
                self.cls.append(cl)

    # Compute proportion of class c
    def pi(self, c):
        if c not in self.cls:
            raise ValueError("Wrong class specified")
        return self.y[self.y == c].shape[0] / self.pts

    # Calculate Node entropy
    def entropy(self):
        res = 0
        for c in self.cls:
            pi_c = self.pi(c)
            if pi_c != 0:
                res -= pi_c * np.log2(pi_c)
        return res

    # Check Node is pure or not
    def is_pure(self):
        for cl in self.cls:
            if self.pi(cl) > 0.99:
                return True
        return False

    # Find the best variable and place to cut
    def split(self):
        min_entropy = 100000000000  # Initial entropy
        split_iloc = -1  # cutting variable
        split_pt = 0  # cutting point
        for col in range(self.X.shape[1]):
            minValue = np.min(self.X.iloc[:, col])
            maxValue = np.max(self.X.iloc[:, col])
            pts = np.random.uniform(minValue, maxValue, 20)  # For random searching for local minimum
            left_Xs = [self.X[self.X.iloc[:, col] < pt] for pt in pts]
            left_ys = [self.y[self.X.iloc[:, col] < pt] for pt in pts]
            right_Xs = [self.X[self.X.iloc[:, col] >= pt] for pt in pts]
            right_ys = [self.y[self.X.iloc[:, col] >= pt] for pt in pts]

            left_subnodes = [Node(left_Xs[i], left_ys[i]) for i in range(len(left_Xs))]
            left_entropy = np.array([node.entropy() * node.pts for node in left_subnodes])
            right_subnodes = [Node(right_Xs[i], right_ys[i]) for i in range(len(right_Xs))]
            right_entropy = np.array([node.entropy() * node.pts for node in right_subnodes])

            sub_entropy = left_entropy + right_entropy
            simMin = np.min(sub_entropy)
            if simMin < min_entropy:
                min_entropy = simMin
                split_iloc = col
                split_pt = pts[np.argmin(sub_entropy)]

        return split_iloc, split_pt
