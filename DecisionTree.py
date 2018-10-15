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
            dist = {}
            for cl in current.cls:
                dist[cl] = current.pi(cl)
            self.tree[n] = [iloc, pt, dist]

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
            dist = {}
            for cl in current.cls:
                dist[cl] = current.pt(cl)
                if current.pi(cl) > max_value:
                    current_type = cl
            self.tree[n] = (current_type, dist)

    # Draw the decision tree
    def draw(self):
        dot = Graph(comment="Decision Tree", format='png')
        for key in self.tree:  # Draw all nodes
            value = self.tree[key]
            if type(value[0]) != str:
                label = "Cut X{0} at {1:.3f}\n".format(value[0] + 1, value[1])
                for cl in value[2]:
                    label += (cl + ": {0:.0%} ".format(value[2][cl]))
                dot.node(str(key), label)
            else:
                label = "Class {}\n".format(value[0])
                for cl in value[1]:
                    label += (cl + ": {} ".format(value[1][cl]))
                dot.node(str(key), label)

        for key in self.tree:  # Draw all edges
            if 2*key in self.tree:
                dot.edge(str(key), str(2 * key), label="<")
                dot.edge(str(key), str(2 * key + 1), label=">=")

        dot.render('tree.gv', view=True)
        print(dot.source)








