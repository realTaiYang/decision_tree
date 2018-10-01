import numpy as np
import pandas as pd
from Node import Node
from graphviz import Graph


class DecisionTree:
    # Constructor
    def __init__(self, X, y):
        self.root = Node(X, y)  # Root node of the tree
        self.tree = {}  # Save nodes position and info in the tree
        self._build_tree(self.root, 1)  # Build tree at instantiation

    # Tree builder. Will be called in constructor
    def _build_tree(self, current, n):
        if current.pts >= 15 and not current.is_pure():  # Continue criterion
            iloc, pt = current.split()
            self.tree[n] = (iloc, pt)
            left_X = current.X[current.X.iloc[:, iloc] < pt]
            left_y = current.y[current.X.iloc[:, iloc] < pt]
            left_child = Node(left_X, left_y)
            right_X = current.X[current.X.iloc[:, iloc] >= pt]
            right_y = current.y[current.X.iloc[:, iloc] >= pt]
            right_child = Node(right_X, right_y)
            self._build_tree(left_child, 2*n)
            self._build_tree(right_child, 2*n+1)
        else:  # Add leaves node (Classes)
            max_value = 0
            for cl in current.cls:
                if current.pi(cl) > max_value:
                    current_type = cl
            self.tree[n] = current_type

    # Draw the decision tree
    def draw(self):
        dot = Graph(comment="Decision Tree", format='png')
        for key in self.tree:  # Draw all nodes
            value = self.tree[key]
            if type(value) != str:
                dot.node(str(key), "Cut X{} at {}".format(value[0] + 1, value[1]))
            else:
                dot.node(str(key), "Class {}".format(value))

        for key in self.tree:  # Draw all edges
            if 2*key in self.tree:
                dot.edge(str(key), str(2 * key))
                dot.edge(str(key), str(2 * key + 1))

        dot.render('tree.gv', view=True)
        print(dot.source)








