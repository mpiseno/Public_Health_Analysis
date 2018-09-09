from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data extraction
data = pd.read_csv('at_risk_queer.csv', sep=',')
X = data.values[:, 0:(data.shape[1] - 2)]
Y = data.values[:, data.shape[1] - 1]

lastrow = data.values[:,-1]
isgay = [x for x in lastrow if x == 1.0]
print(len(isgay))
percentgay = len(isgay) / len(lastrow)
print(percentgay)