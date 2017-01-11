import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from sklearn import linear_model

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

# train['Age'].hist()
# P.show()

# The titanic variable is available here.
train["Age"] = train["Age"].fillna(train["Age"].median())

# Replace all the occurences of male with the number 0.
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

train["Embarked"] = train["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

# Train 1

t1 = train[['Pclass', 'Sex', 'Age', 'Embarked', 'Survived']]
X = t1[['Pclass', 'Sex', 'Age', 'Embarked']]
Y = t1[['Survived']]

print(t1)

print(t1.dtypes)

regr = linear_model.LinearRegression()
regr.fit(X[0:840], Y[0:840])

print(regr)
print(regr.coef_)

m = np.mean((regr.predict(X[841:])-Y[841:])**2)
print(m)

score = regr.score(X[841:], Y[841:])
print(score)