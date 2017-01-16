import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
import re
from data_preporation import prepare

# Print you can execute arbitrary python code
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

# Print to standard output, and see the results in the "log" section below after running your script
print("\n\nLen of the training data: " + str(len(train)))
print("\n\nLen of the test data: " + str(len(test)))
print("\n\nTypes of the data: ")
print(train.dtypes)
print("\n\nTop of the training data:")
print(train.head())
print("\n\nSummary statistics of training data")
print(train.describe())
print("----------------------------")

prepare(train)

# Get all the titles and print how often each one occurs.
# print(pd.value_counts(train["Cabin"]))

print(train['Pclass'] == 1)