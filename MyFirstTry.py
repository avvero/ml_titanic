import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import linear_model
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

# train['Age'].hist()
# P.show()

prepare(train)

# Train 1
predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'NameLength', 'Title']
alg = linear_model.LogisticRegression()
scores = cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=3)

print("Scores")
print(scores.mean())

print("-------TEST--------")

prepare(test)

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