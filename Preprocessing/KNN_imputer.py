#Python program to fill the missing value in dataset using KNNImputer
#Credits to Alan Chang as the consultant on development of the code

#Import modules
import numpy as np
import os
import pandas as pd
from sklearn.impute import KNNImputer

os.chdir('D:/Car Care (Final Project)/Preprocessing/NewData/')
fileloc = os.listdir()

#Open dataset
for csv_file in fileloc:
    df = pd.read_csv(csv_file, index_col = [0])

#Replace 0 with NaN val
    df.replace(0, np.nan, inplace = True)
#Count number of missing value
    print(csv_file,"Number of Null Value:\n", df.isnull().sum())

#KNN imputer
    imputer = KNNImputer(n_neighbors = 5)
    df_filled = imputer.fit_transform(df)

#Store imputed data back to dataframe
    df = pd.DataFrame(df_filled, columns = df.columns, index = df.index)
    print(csv_file,"Number of Null Value:\n", df.isnull().sum())
    df.reset_index(inplace = True)

    dupl = df.duplicated(subset = 'time').sum()
    print("Number of Duplicated Value in {}:".format(csv_file), dupl)
    print('Shape of Dataset:', df.shape)
    del_dup = df.drop_duplicates(subset = 'time',keep = 'first', inplace = True)
    print("Number of Duplicated Value in {}:".format(csv_file), df.duplicated(subset = 'time').sum())
    print('Shape of Dataset:', df.shape)

#Export file to csv
    newloc = 'D:/Car Care (Final Project)/Preprocessing/NewData/'
    newfile = newloc + csv_file
    df.to_csv(newfile, index = True)
print('Imputting Done\nThank You')