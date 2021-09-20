#Python program to predict values of Insurance Fee

#Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Open dataset
df = pd.read_csv('Machine_Learning/Car-Care_Dataset.csv')

#Show statistical summary of dataset
print(df.describe())

#Split dataset
x = df.drop(['time', 'Insurance Fee'], axis = 1)
y = df['Insurance Fee']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 50)

#Standardize dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Select ML algorithm
model = LinearRegression()
model.fit(x_train, y_train)

#Retrieve Intercept and Coefficient
print('Intercept Value: ', model.intercept_)
print('Coefficient Value:\n', model.coef_)

#Make prediction
y_pred = model.predict(x_test)
#print('Predicted Result:\n', y_pred)

#Save finalized model
with open("D:/Car Care (Final Project)/Model/Finalized_UBI_LinearRegression_Model.pkl", 'wb') as UBI_LR:
    pickle.dump(model, UBI_LR)

#Compare actual value with predicted value
data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(data)

# Visualize the comparison results.
df1 = data.head(30)
df1.plot(kind='bar', figsize=(10, 8))
plt.title('Actual vs Predicted')
plt.xlabel('Driving Trip Number')
plt.ylabel('Predicted Insurance Fee')
plt.grid(which='major', linestyle='-', linewidth='0.5', color= 'black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color= 'green')
plt.show()

#Evaluate model
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
print('Mean Absolute Error (MAE) Value: ', mae)
print('Mean Squared Error (MSE) Value: ', mse)
print('Root Mean Squared Error (RMSE) Value: ', rmse)
print('R2 Score: ', r2)