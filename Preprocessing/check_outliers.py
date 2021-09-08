#Python program to check outliers

#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

#Open dataset
df = pd.read_csv('Preprocessing/NewData/UserDataset.csv', index_col = [1])
#print(df.columns)
df.drop(df.columns[df.columns.str.contains('Unnamed')], axis = 1, inplace = True)

#Create a box plot to see check any outliers
df.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False)
plt.show()

df.to_csv('Preprocessing/NewData/CarCare_Dataset.csv', index = False)