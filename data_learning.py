import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
import re
from data_preporation import prepare, get_digits_only, skipbig
from data_preporation import get_last_name
from data_preporation import get_family_id
import operator

# Print you can execute arbitrary python code
train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

# Get all the titles and print how often each one occurs.
# print(pd.value_counts(train["Cabin"]))

# print(pd.value_counts(train['Cabin']))

#print(train['Name'].apply(get_last_name))


print(train['Cabin'].unique())
