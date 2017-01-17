import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def find():
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic[predictors], titanic["Survived"])

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)

    # Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    # Pick only the four best features.
    predictors = ["Pclass", "Sex", "Fare", "Title"]

    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
    # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())
    return ""