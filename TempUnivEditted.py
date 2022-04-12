
"""
Created on Wed Sep 15 21:59:20 2021

LSTM model for temperature prediction (enhanced)

@author: nawafiQbal
"""

# Part -1 Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the training set
data = pd.read_csv("Temp_3000.csv")     
new_data = data.dropna(axis = 0, how ='any')
# Selecting the desired rows and cols
dataset = new_data.iloc[:, 1:2].values

# line plot for the database
data.plot()
plt.savefig("X.png", figsize=(10,5))
plt.show()

# Feature Scaling
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Dividing the dataset into Training, validation and Testing sets as 80-10-10%, respectively
train_size = int(len(dataset) * 0.8)
validation_size = int(len(dataset)* 0.1)
test_size = len(dataset) - train_size - validation_size
train = dataset[0:train_size] 
validation = dataset[train_size:train_size + validation_size]
test = dataset[train_size + validation_size:]
print(len(train), len(validation), len(test))

# Converting an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    dataX.append(dataset[i:(i+look_back), 0])
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

look_back = 24
trainX, trainY = create_dataset(train, look_back=look_back)
validationX, validationY = create_dataset(validation, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)

# Reshaping input to be [samples, time steps, features] to fire the RNN model
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
validationX = np.reshape(validationX, (validationX.shape[0], validationX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Part 2 - Building the RNN (LSTM)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf 

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(50,  return_sequences = True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(1))

# Compiling the RNN
model.compile(loss='mean_squared_error', optimizer='adam')

# Adding the hyperparameter early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience=4)

# Computing the time for building the model
import timeit

start = timeit.default_timer()

# Fitting the RNN to the Training set
model.fit(trainX, trainY, epochs=10, batch_size=30, callbacks=[early_stopping], validation_data=(validationX, validationY), verbose=2)

stop = timeit.default_timer()

print('Training Time: ', stop - start) 

# Part 3 - Making the predictions and visualising the results
import timeit

start = timeit.default_timer()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

stop = timeit.default_timer()

print('Prediction Time: ', stop - start) 

# Inverting the predictions before calculating error so that reports will be in same units as our original data
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Finding the RMSE (Error rate)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
