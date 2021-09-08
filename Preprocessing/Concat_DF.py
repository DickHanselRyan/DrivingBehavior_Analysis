#Python program to concatenate all datasets

#Import modules
import glob
import pandas as pd

file_loc = glob.glob("Preprocessing/NewData/*.csv")

csv_list = list()
for csv_file in file_loc:    
    df = pd.read_csv(csv_file, header = 0, index_col = [1])
    csv_list.append(df)

new_df = pd.concat(csv_list, axis = 0, ignore_index = False)

new_df.to_csv('D:/Car Care (Final Project)/Preprocessing/NewData/UserDataset.csv', index = True)
print("Processing Done\nThank You")