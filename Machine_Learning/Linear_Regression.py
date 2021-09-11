#Python program to predict distance travelled using n-value of fuel

#Import modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Open dataset
df = pd.read_csv('Machine_Learning/Car-Care_Dataset.csv')

#Show dimension of dataset, statistical summary, some top rows of data, and columns in dataset
#print(df.columns, '\n', df.describe(), '\n', df.shape, '\n', df.head(20))

#Summarize distribution of attributes
#pd.set_option('precision', 1)
#print(df.describe())

#Show correlation between numeric attributes
#pd.set_option('precision', 2)
#print(df.corr(method = 'pearson'))

#Exploratory Data Analysis (EDA)
#Univariate plot
#df.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False, title = 'Box and Whiskers Plot')
#plt.show()

#Show heatmap of feature correlation
#dfcorr = df.corr()
#sb.heatmap(dfcorr, cmap = 'YlGnBu', annot = True)
#plt.title("Correlation Heatmap")
#plt.tight_layout()
#plt.show()

#Creating ML Model
#Split dataset
x = df['Fuel used (L)'].values.reshape(-1, 1)
y = df['Distance travelled (km)'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

#Plot scatter of data before model predicts
plt.scatter(x_test, y_test, color = 'green')
plt.title('Raw Data')
plt.xlabel('Fuel Used (L)')
plt.ylabel('Distance Travelled (km)')
plt.show()

#Standardize the dataset
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Select ML algorithm
model = LinearRegression()
model.fit(x_train, y_train)

#Save Model with pickle
with open("D:/Car Care (Final Project)/Model/Finalized_LR_Model.pkl", 'wb') as q:
    pickle.dump(model, q)

#Retrieve Intercept and Coefficient
print('Intercept Value: ', model.intercept_)
print('Coefficient Value:\n', model.coef_)

#Make prediction
y_pred = model.predict(x_test)
#print('Predicted Result:\n', y_pred)

#Compare actual value with predicted value
data = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(data)

#Show predicted result
plt.scatter(data['Actual'], data['Predicted'], color = 'teal')
plt.plot(data['Predicted'], data['Predicted'], color = 'red', linewidth = 2)
plt.title('Distance Travelled Predicted')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()

# Visualize the comparison results.
df1 = data.head(30)
df1.plot(kind='bar', figsize=(10, 8))
plt.title('Actual vs Predicted')
plt.xlabel('Driving Trip Number')
plt.ylabel('Driving Distance (km)')
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