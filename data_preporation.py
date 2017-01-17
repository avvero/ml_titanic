import re
import numpy as np

# A function to get the title from a name.
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# A function to get the title from a name.
def get_last_name(name):
    title_search = re.search('([A-Z])\w+', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(0)
    return np.NaN

def get_family_id(row, map):
    last_name = get_last_name(row['Name'])
    if last_name not in map:
        if len(map) == 0:
            current_id = 1
        else:
            current_id = len(map) + 1
        map[last_name] = current_id
    return ""

def set_family_id(row, map):
    familyId = -1
    if row["FamilySize"] > 0:
        familyId = map[get_last_name(row['Name'])]
    return familyId

def prepare(train, family_id_mapping):
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

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {
        "Mr": 1,
        "Miss": 2,
        "Mrs": 3,
        "Master": 4,
        "Dr": 5,
        "Rev": 6,
        "Major": 7,
        "Col": 8,
        "Mlle": 9,
        "Mme": 10,
        "Don": 11,
        "Lady": 12,
        "Countess": 13,
        "Jonkheer": 14,
        "Sir": 15,
        "Capt": 16,
        "Ms": 17,
        "Dona": 18
    }
    for k, v in title_mapping.items():
        titles[titles == k] = v

    # Add in the title column.
    train["Title"] = titles

    train["FamilyId"] = train.apply(lambda row: set_family_id(row, family_id_mapping), axis=1)

    return ""
