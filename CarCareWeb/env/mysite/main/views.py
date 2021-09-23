# Create your views here.
import io
from django.shortcuts import render, redirect
from pymongo import collection
from .utils import get_db_handle, get_collection_handle
from django.contrib import messages #import messages
from django.contrib.auth import login, authenticate, logout #import authenticate
from django.contrib.auth.forms import AuthenticationForm #import AuthenticationForm
from .forms import * 
from .models import *
import pandas as pd
import glob, os
from os import listdir
import json
import pymongo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import base64, urllib
from io import BytesIO

db_handle, mongo_client = get_db_handle('CarCare_DB', 'localhost', 27017, 'USERNAME', 'PASSWORD')
collection_handle = get_collection_handle(db_handle, 'REGIONS_COLLECTION')
conn = pymongo.MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false")
db = conn["CarCare_DB"]

def homepage(request):
	return render(request = request, template_name="./home.html")

def about_us(request):
	return render(request = request, template_name="main/aboutus.html"	)

def contact_us(request):
	context = {}
	if request.method =='POST':
		form = ContactForm(request.POST)
		if form.is_valid():
			form.save()
			messages.success(request, "Sumbit successful." )
	form = ContactForm()
	context = {'form':form }
	return render(request = request, template_name="main/contact_us.html")

def register(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect("main:homepage")
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm
	return render (request=request, template_name="main/register.html", context={"form":form})

def login_request(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect("main:homepage")
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="main/login.html", context={"form":form})

def logout_request(request):
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return redirect("main:homepage")

def userpage(request):

	return render(request=request, template_name="main/user.html")

def upload_file(request):
	context = {}

	if request.method == 'POST':
		form = UploadFileForm(request.POST,request.FILES)
		if form.is_valid():
			title = form.cleaned_data['title']
			file = form.cleaned_data['file']

			uploadfile = UploadFile()
			uploadfile.title = title
			uploadfile.file = file
			uploadfile.save()

		messages.info(request, "You have successfully uplaod file.")
		files = UploadFile.objects.values()
		files_pandas = pd.DataFrame(files)

	else:
		form = UploadFileForm()
	context['form'] = form
	
	return render(request,'main/upload.html',context={'form':form})

def data(request):
	files_path = 'C:/Users/USER/Desktop/code/env/mysite/static/files/filesupload'
	read_files = glob.glob(os.path.join(files_path,"*.csv"))

	np_array_values = []
	for files in read_files:
		data = pd.read_csv(files)
		np_array_values.append(data)
		print(files)

	with open("C:/Users/USER/Desktop/code/env/mysite/main/data/info.json","r") as file:
		JsonFile = json.load(file)
		client = pymongo.MongoClient()
		database = client[JsonFile["database"]]
		collection = database[JsonFile["collection"]]
		files = listdir(JsonFile["filePath"])

		for file in range(len(files)):

			fileName = JsonFile["filePath"]+"\\"+files[file]
			csvFile = pd.read_csv(fileName)
    
			[x,y] = csvFile.shape
			columns = list(csvFile.columns)
			data = csvFile.values
    
			for row in range(x):
				dataRow = data[row]
				DataDict = dict(zip(columns, dataRow))
				collection.insert(DataDict)
				
			print("Data has been inserted")
			# delete already read files
			os.remove(os.path.join(files_path, fileName))
		
	col = db["main_file"]
	if request.method == "POST":
		startdate = request.POST.get('startdate')
		enddate = request.POST.get('enddate')
		result = col.find({'date_time':{'$gte' : startdate, '$lte' : enddate}})
		return render(request,'main/data.html',{"data":result})
	else:
		file = File.objects.all()
	return render(request, 'main/data.html',{"data":file})

def data_analysis(request):
	col = db["main_file"]
	data = col.find()
	data = list(data)

	df = pd.DataFrame(data)
	#Creating ML Model
	#Split dataset
	x = df['Fuel_used'].values.reshape(-1, 1)
	y = df['Distance_travelled'].values.reshape(-1, 1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

	#Plot scatter of data before model predicts
	plt.scatter(x_test, y_test, color = 'green')
	plt.title('Raw Data')
	plt.xlabel('Fuel Used (L)')
	plt.ylabel('Distance Travelled (km)')
	#plt.show()
	#plt.close()
	plt.legend()
	fig = plt.gcf()
	buf = io.BytesIO()

	fig.savefig(buf, format='png')
	buf.seek(0)
	string = base64.b64encode(buf.read())
	uri = urllib.parse.quote(string)

	#Standardize the dataset
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)

	#Select ML algorithm
	model = LinearRegression()
	model.fit(x_train, y_train)

	#Save Model with pickle
	with open("C:/Users/USER/Desktop/code/env/mysite/main/Model/Finalized_LR_Model.pkl", 'wb') as q:
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
	#plt.show()
	#plt.close()
	plt.legend()
	fig = plt.gcf()
	buf = io.BytesIO()

	fig.savefig(buf, format='png')
	buf.seek(0)
	string = base64.b64encode(buf.read())
	uri = urllib.parse.quote(string)

	# Visualize the comparison results.
	df1 = data.head(30)
	df1.plot(kind='bar', figsize=(10, 8))
	plt.title('Actual vs Predicted')
	plt.xlabel('Driving Trip Number')
	plt.ylabel('Driving Distance (km)')
	plt.grid(which='major', linestyle='-', linewidth='0.5', color= 'black')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color= 'green')
	#plt.show()
	#plt.close()
	plt.legend()
	fig = plt.gcf()
	buf = io.BytesIO()

	fig.savefig(buf, format='png')
	buf.seek(0)
	string = base64.b64encode(buf.read())
	uri = urllib.parse.quote(string)
	#Evaluate model
	mae = metrics.mean_absolute_error(y_test, y_pred)
	mse = metrics.mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
	r2 = metrics.r2_score(y_test, y_pred)
	print('Mean Absolute Error (MAE) Value: ', mae)
	print('Mean Squared Error (MSE) Value: ', mse)
	print('Root Mean Squared Error (RMSE) Value: ', rmse)
	print('R2 Score: ', r2)

	return render(request, "main/data_analysis.html",{'data':uri})

def data_prediction(request):
	return render(request = request, template_name="main/data_prediction.html")	



