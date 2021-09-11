#Python classification module using NaiveBayes

#Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

#Open dataset
df = pd.read_csv('./Machine_Learning/Car-Care_Dataset.csv')
#print(df.head(20), df.dtypes)

#Select independent and dependent variable(s)
x = df.drop(['time', 'Average speed (km/h)', 'Fuel used (L)', 'Driver Type', 'Insurance Fee'], axis = 1)
#print(x.head(30))
y = df['Driver Type']
#print(y.head(30))

#Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 500)
print('Shape of x_train: ', x_train.shape, 'Shape of x_test: ', x_test.shape)

#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Select ML algorithm
classifier = OneVsRestClassifier(GaussianNB())
classifier.fit(x_train, y_train)

#Save finalized model
with open("D:/Car Care (Final Project)/Model/Finalized_NB_Model.pkl", 'wb') as model:
    pickle.dump(classifier, model)

#Predict result
y_pred = classifier.predict(x_test)
#Get probability
y_pred_prob = classifier.predict_proba(x_test)

#Accuracy score
score = metrics.accuracy_score(y_test, y_pred)
print('Accuracy of Naive Bayes Classifier: %.2f' % score)

#Classification Report
cr = metrics.classification_report(y_test, y_pred)
print("Classification Report of the Model:\n", cr)

#Confusion Matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix of the Model:\n", conf_mat)

#Show Confusion Matrix Heatmap
plt.title("Confusion Matrix's Heatmap")
sb.heatmap(conf_mat, annot = True, fmt ='d', cmap = 'YlGnBu')
plt.xlabel("Actual Classification")
plt.ylabel("Predicted Classification")
plt.show()

#ROC curve for classes
fpr = {}
tpr = {}
threshold = {}
#There are 3 classes of classification (0, 1, 2)
n_class = 3

for i in range(n_class):
    fpr[i], tpr[i], threshold[i] = metrics.roc_curve(y_test, y_pred_prob[:, i], pos_label = i)

#Plot the curve
plt.plot(fpr[0], tpr[0], linestyle = '--', color = 'blue', label = 'Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle = '--', color = 'green', label = 'Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle = '--', color = 'orange', label = 'Class 2 vs Rest')
plt.title('Multiclass ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'best')
plt.show()