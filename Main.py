import numpy as np
import pandas as pd
import DecisionTree
from DecisionTree import DecisionTree


data = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                   header=None)
data = data.iloc[:, :13]

X = data.iloc[:, 3:13]
y = data[1]

test = DecisionTree(X, y)
test.draw()
