import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from DecisionTree import DecisionTree


data = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                   header=None)
data = data.iloc[:, :12]

X = data.iloc[:, 2:12]
y = data[1]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
print(clf.score(X, y))

test = DecisionTree(X, y)
test.draw()
