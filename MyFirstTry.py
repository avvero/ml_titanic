import numpy as np
import pandas as pd
from sklearn import cross_validation
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
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
# train["SibSp"] = train["SibSp"].fillna(0)
# train["Parch"] = train["Parch"].fillna(0)

# Train 1

t1 = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Survived']]
X = t1[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
Y = t1[['Survived']]

predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
alg = linear_model.LogisticRegression()
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

print("----------------")

print(scores.mean())


print("-------------------")
print("-------TEST--------")
print("-------------------")

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

# Train the algorithm using all the training data
alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv", index=False)