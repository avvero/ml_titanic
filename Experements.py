import numpy as np
import pandas as pd
from sklearn import linear_model

a = np.array([[1, 1], [2, 2], [3, 3]])

print(a.shape)
print(a[0:2, 0:2])

f = pd.DataFrame(a, index=a[:,0])
print(f)