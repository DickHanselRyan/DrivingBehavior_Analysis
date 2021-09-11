#Create a New Output to CSV
#To Classify Driver

import pandas as pd
import numpy as np

df = pd.read_csv('ML_Code/CarCare_Dataset.csv')

#Categorize Driver Based on the Type
#1 is Safe
#2 is Moderate
#3 is Dangerous
conditions = [
    (df['Vehicle speed (km/h)'] >= 0) & (df['Vehicle speed (km/h)'] <= 60),
    (df['Vehicle speed (km/h)'] > 60) & (df['Vehicle speed (km/h)'] <= 80),
    (df['Vehicle speed (km/h)'] > 80)
    ]
values = [1, 2, 3]

df['Driver Type'] = np.select(conditions, values)
df.to_csv('D:/Car Care (Final Project)/Preprocessing/NewData/Classified_User.csv', index = False)
