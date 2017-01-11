import numpy as np
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt

#Print you can execute arbitrary python code
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

print(train.head())