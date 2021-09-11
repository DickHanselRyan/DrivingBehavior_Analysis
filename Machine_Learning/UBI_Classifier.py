#Create a new column on CSV file
#To classify drivers' UBI premium

#Import modules
import pandas as pd
import numpy as np

df = pd.read_csv('ML_Code/Classified_User.csv')

#Categorize UBI discount rate on some conditions
#Insurance default price per year is NT$1300
#0-5km = 60% discount
#6-10km = 30% discount
#10km++ = 10% discount

#Count discount
count1 = int(1300 * 0.6)
count2 = int(1300 * 0.3)
count3 = int(1300 * 0.1)

#Count total price after discount
newval1 = 1300 - count1
newval2 = 1300 - count2
newval3 = 1300 - count3

conditions = [
    (df['Distance travelled (km)'] >= 0) & (df['Distance travelled (km)'] <= 6),
    (df['Distance travelled (km)'] >= 6) & (df['Distance travelled (km)'] <= 10),
    (df['Distance travelled (km)'] >= 10)]

values = [newval1, newval2, newval3]
df['Insurance Fee'] = np.select(conditions, values)
df.to_csv('D:/Car Care (Final Project)/ML_Code/Car-Care_Dataset.csv', index = False)
print("Processing Done\nThank You")