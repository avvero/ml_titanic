import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import linear_model
from data_preporation import prepare
from data_preporation import get_family_id
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from utils import plot_learning_curve
import matplotlib.pyplot as plt

# Print you can execute arbitrary python code
train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

# Family distribution
family_id_mapping = {}
train.apply(lambda row: get_family_id(row, family_id_mapping), axis=1)
test.apply(lambda row: get_family_id(row, family_id_mapping), axis=1)

prepare(train, family_id_mapping)

# Train 1
predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'NameLength', 'Title', 'FamilyId']

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
print(scores.mean())

print("Alg")
print(alg)

print("Plot")
plot_learning_curve(pipeline, "sdf", train[predictors], train["Survived"], (-0.1, 1.1), cv=3, n_jobs=1)
plt.show()

print("-------TEST--------")

prepare(test, family_id_mapping)

# Train the algorithm using all the training data
alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("data/kaggle.csv", index=False)
