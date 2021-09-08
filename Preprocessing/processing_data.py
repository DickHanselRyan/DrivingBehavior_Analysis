#Python program to process raw dataset

#Import modules
import os
import pandas as pd

#Locate dataset
os.chdir('D:/Car Care (Final Project)/Preprocessing/Data/')
fileloc = os.listdir()
for csv_file in fileloc:
    df = pd.read_csv(csv_file)
    
#Drop columns called unnamed
    df.drop(df.columns[df.columns.str.contains('Unnamed')], axis = 1, inplace = True)
    #print(csv_file, df.columns, '\n')

#Check unique value of each column
    uniq_val = df.nunique()
    #print(csv_file, 'Unique Value per column:\n', uniq_val, '\n')

#Remove columns with less unique value
    for col in df.columns:
        if len(df[col].unique()) <= 7:
            df.drop(col, axis = 1, inplace = True)
        #print(csv_file, df.columns)

#Remove unnecessary columns
    df.drop(df[['Average fuel consumption (L/100km)', 'Average fuel consumption (total) (L/100km)', 'Calculated boost (bar)',
                'Distance travelled (total) (km)', 'Calculated instant fuel consumption (L/100km)',
                'Calculated instant fuel rate (L/h)', 'Fuel used price ($)', 'Fuel used (total) (L)', 
                'Intake manifold absolute pressure (kPa)', 'Instant engine power (based on fuel consumption) (hp)',
                'Throttle position (%)', 'Vehicle acceleration (g)']], axis = 1, inplace = True)
    #print(csv_file, df.columns)

#Change the type of time_index to a datetime type
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].apply(lambda x: x.replace(second = 0, microsecond = 0))
    df['time'] = df['time'].dt.strftime('%H:%M')
    #print(csv_file, df['time'].head())

#Check number of duplicate value on time column
    print(csv_file, df.duplicated().sum())
    print(csv_file, "Shape's Before Duplicate Value Deleted:", df.shape)

#Delete duplicate value
    del_dup = df.drop_duplicates(keep = False, inplace = True)
    print(csv_file, "Shape's After Duplicate Value Deleted:", df.shape, '\n')

#Set new index
    df.set_index('time', inplace = True)

#Export to a new folder
    newloc = 'D:/Car Care (Final Project)/Preprocessing/NewData/'
    newfile = newloc + csv_file
    df.to_csv(newfile, index = True)
print('Processing Done\nThank You')
