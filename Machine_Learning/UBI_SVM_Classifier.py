#Python program to classify Insurance Fee
#Using SVM ML model

#Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#Load Dataset
df = pd.read_csv('./Machine_Learning/Car-Care_Dataset.csv')
#print(df.describe())

#Split dataset
x = df.drop(['time', 'Insurance Fee'], axis = 1)
y = df['Insurance Fee']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 1000)

#Feature Scaling
#Standardize dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Select ML algorithm
model = SVC()
model.fit(x_train, y_train)

#Save model
with open("D:/Car Care (Final Project)/Model/Finalized_SVM_Model.pkl", 'wb') as SVM_class:
    pickle.dump(model, SVM_class)

#Predict result
y_pred = model.predict(x_test)

#Count accuracy of model
score = sm.accuracy_score(y_test, y_pred)
print("Accuracy Score of Classifier: %.2f" % score)

#Classification Report
cr = sm.classification_report(y_test, y_pred)
print("Classification Report of the Model:\n", cr)

#Confusion matrix
conf_mat = sm.confusion_matrix(y_test, y_pred)
print("Confusion Matrix of the Model:\n", conf_mat)

#Plot confusion matrix
plt.title("SVM Confusion Matrix's Heatmap")
sb.heatmap(conf_mat, annot = True, fmt ='d', cmap = 'YlGnBu')
plt.xlabel("Actual Classification")
plt.ylabel("Predicted Classification")
plt.show()