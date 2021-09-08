#Python program to merge rows in dataset
#Credits to Alan Chang as the consultant on development of the code

#Import modules
import csv
import glob
import os
import re
import pandas as pd

#Locate dataset
filepaths = glob.glob("Preprocessing/NewData/*.csv")

#Open dataset
for csv_file in filepaths:
    csvArray = list()
    newcsv = list()
    with open(csv_file, encoding = "utf-8") as tempFile:
        readfile = csv.reader(tempFile)
#Store rows / values in dataset to a row
        for row in readfile:
            csvArray.append(row)
    
#Remove header / column name from dataset
    header_storage = csvArray[0]
    del csvArray[0]
    del csvArray[1]
    del csvArray[2]
    csvLen = len(csvArray)
    csvCheck = csvLen % 4

#Trim data for standardization
    if csvCheck == 1:
        del csvArray[-1]
    elif csvCheck == 2:
        del csvArray[-1]
        del csvArray[-2]
    elif csvCheck == 3:
        del csvArray[-1]
        del csvArray[-2]
        del csvArray[-3]
    else:
        pass
    
    csvLen = len(csvArray)
    csvCheck = csvLen % 4
    #print(csvCheck)

#Merge rows
    for rows in range(0, len(csvArray), 4):
        tmp0 = csvArray[rows]
        tmp1 = csvArray[rows + 1]
        tmp2 = csvArray[rows + 2]
        tmp3 = csvArray[rows + 3]

        for idx, item in enumerate(tmp0):
            if item == "":
                if tmp1[idx] != "":
                    tmp0[idx] = tmp1[idx]
                elif tmp2[idx] != "":
                    tmp0[idx] = tmp2[idx]
                elif tmp3[idx] != "":
                    tmp0[idx] = tmp3[idx]
                else:
                    pass
        newcsv.append(tmp0)

#Convert to DF
    df = pd.DataFrame(newcsv, columns = header_storage)
    df.set_index('time', inplace = True)

#Save the new dataset
    originalName = str(os.path.basename(csv_file))
    newname = re.sub("Processed ", "Cleaned ", originalName)
    newPath = "Preprocessing/NewData/" + newname
    df.to_csv(newPath, index = True)
print('Merging Done\nThank You')