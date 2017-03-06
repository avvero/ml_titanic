import numpy as np
import pandas as pd

# Print you can execute arbitrary python code
train = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

#print(train)
#print(train.describe())

# Get all the titles and print how often each one occurs.
# print(pd.value_counts(train["Cabin"]))

# print(pd.value_counts(train['Cabin']))

print(train['Ticket'].unique())