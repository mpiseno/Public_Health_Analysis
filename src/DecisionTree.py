from sklearn import tree
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

'''
Data Extraction
'''

data = pd.read_csv('at_risk_queer.csv', sep=',')
X = data.values[:, 0:(data.shape[1] - 2)]
Y = data.values[:, data.shape[1] - 1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


'''
Classification
'''

clf = tree.DecisionTreeClassifier(max_depth=6)
clf.fit(X_train, Y_train)

print("Accuracy on the train set: {:.3f}".format(clf.score(X_train, Y_train)))
print("Accuracy on the test set: {:.3f}".format(clf.score(X_test, Y_test)))