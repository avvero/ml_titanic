import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation

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

print("Train SET")
print(t1)

regr = linear_model.LogisticRegression()
regr.fit(X[0:840], Y[0:840])

print("Result")
print(regr)
print(regr.coef_)

print("----------------")

print("Predict")
P = regr.predict(X[841:])
Pf = pd.DataFrame(P, columns = ['Survived'])

Vf = Y[841:].reset_index(drop=True)

print("m = " + str(np.mean(Pf - Vf)))

score = regr.score(X[841:], Y[841:])
print("score = " + str(score))


print("-------------------")
print("-------TEST--------")
print("-------------------")

test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

TP = regr.predict(test[['Pclass', 'Sex', 'Age', 'Embarked']])
TPf = pd.DataFrame(TP, columns = ['Survived'])
print("Predictions")

result = pd.concat([test[['PassengerId']], TPf],  axis=1)

print(np.size(result))

result.to_csv("kaggle.csv", index=False)