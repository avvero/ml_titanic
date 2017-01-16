import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import linear_model
from data_preporation import prepare
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from titanic_support import plot_learning_curve
import matplotlib.pyplot as plt


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

polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
alg = linear_model.LogisticRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("logistic_regression", alg)])
scores = cross_val_score(
    pipeline,
    train[predictors],
    train["Survived"],
    cv=3,
    # scoring="neg_mean_squared_error"
)

print("Scores")
print(scores)
print(scores)
print(scores.mean())

print("Alg")
print(alg)

print("Plot")
plot_learning_curve(pipeline, "sdf", train[predictors], train["Survived"], (0.7, 1.01), cv=3, n_jobs=1)
plt.show()

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