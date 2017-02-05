import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import linear_model
from data_preporation import prepare, get_cabin_id
from data_preporation import get_family_id
from data_preporation import get_ticket_prefix_id
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from utils import plot_learning_curve
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

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

print(get_ticket_prefix_id_mapping)

print("-------KMEANS--------")

random_state = 170

KMF = ['Ticket_s', 'Pclass']
kmx = train[KMF]

km_model = KMeans(n_clusters=2, random_state=random_state).fit(kmx)
print(km_model.score(kmx))
y_pred = km_model.predict(kmx)
print(y_pred)

plt.figure(figsize=(12, 12))
plt.scatter(train[['Ticket_s']], train[['Pclass']], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")
plt.show()

train['Ticket_s_g'] = y_pred

# Train 1
predictors = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'NameLength', 'Title',
              'FamilyId', 'Ticket_s_g', 'CabinN']

# Perform feature selection
if True:
    selector = SelectKBest(f_classif, k=5)
    selector.fit(train[predictors], train["Survived"])

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)

    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

print("------- Learn --------")

polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
#alg = linear_model.LogisticRegression()
alg = RandomForestClassifier(n_estimators=100)
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("logistic_regression", alg)])
scores = cross_val_score(
    pipeline,
    train[predictors],
    train["Survived"],
    cv=3
)

print("Scores")
print(scores)
print(scores.mean())

print("Alg")
print(alg)

print("------- Bias-Variance --------")

print("Plot")
plot_learning_curve(pipeline, "sdf", train[predictors], train["Survived"], (-0.1, 1.1), cv=3, n_jobs=1)
plt.show()

print("-------TEST--------")

prepare(test, family_id_mapping, get_ticket_prefix_id_mapping, cabin_id_mapping)

test['Ticket_s_g'] = km_model.predict(test[KMF])

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