from titanic_support import get_title


def prepare(train):
    # The titanic variable is available here.
    train["Age"] = train["Age"].fillna(train["Age"].median())
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Embarked"] = train["Embarked"].fillna("S")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    train["SibSp"] = train["SibSp"].fillna(0)
    train["Parch"] = train["Parch"].fillna(0)
    # Generating a familysize column
    train["FamilySize"] = train["SibSp"] + train["Parch"]
    # The .apply method generates a new series
    train["NameLength"] = train["Name"].apply(lambda x: len(x))
    # Get all the titles and print how often each one occurs.
    titles = train["Name"].apply(get_title)

    return ""
