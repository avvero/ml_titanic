import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from data_preporation import get_cabin_id
from data_preporation import get_family_id
from data_preporation import get_ticket_prefix_id
from data_preporation import prepare
from utils import plot_confusion_matrix
from utils import plot_learning_curve

# Print you can execute arbitrary python code
train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

# Family distribution
family_id_mapping = {}
train.apply(lambda row: get_family_id(row, family_id_mapping), axis=1)
test.apply(lambda row: get_family_id(row, family_id_mapping), axis=1)
# Ticket distribution
get_ticket_prefix_id_mapping = {}
train.apply(lambda row: get_ticket_prefix_id(row, get_ticket_prefix_id_mapping), axis=1)
test.apply(lambda row: get_ticket_prefix_id(row, get_ticket_prefix_id_mapping), axis=1)
# cabin distribution
cabin_id_mapping = {}
train.apply(lambda row: get_cabin_id(row, cabin_id_mapping), axis=1)
test.apply(lambda row: get_cabin_id(row, cabin_id_mapping), axis=1)

prepare(train, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping)

# Train 1
predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'NameLength', 'Title',
              'FamilyId', 'CabinN']

# Perform feature selection
if False:
    selector = SelectKBest(f_classif, k=5)
    selector.fit(train[predictors], train["Survived"])

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)

    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

print("------- Learn --------")

from sklearn.model_selection import ShuffleSplit

polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
alg = linear_model.LogisticRegression()
# alg = AdaBoostClassifier()
# alg = RandomForestClassifier(n_estimators=300)
# alg = SVC()
# alg = MLPClassifier(hidden_layer_sizes=(24,24,24))

pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("logistic_regression", alg)])
scores = cross_val_score(
    pipeline,
    train[predictors],
    train["Survived"],
    cv=ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
)
print(scores)
print(scores.mean())

print("Alg")
print(alg)

print("------- Bias-Variance --------")

from sklearn.metrics import confusion_matrix

# Train the algorithm using all the training data

alg.fit(train[predictors], train["Survived"])
cnf_matrix = confusion_matrix(train["Survived"], alg.predict(train[predictors]))
plot_confusion_matrix(cnf_matrix, classes=[], title='Confusion matrix, without normalization')
plt.show()

print("------- Bias-Variance --------")

print("Plot")
plot_learning_curve(pipeline, "sdf", train[predictors], train["Survived"], (-0.1, 1.1), cv=3, n_jobs=1)
plt.show()

print("-------TEST--------")

# TEST
prepare(test, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping)

# Train the algorithm using all the training data
# alg.fit(train[predictors], train["Survived"])

# Make predictions using the test set.
predictions = alg.predict(test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("data/kaggle.csv", index=False)
