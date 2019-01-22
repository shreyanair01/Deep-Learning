#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:13:08 2019

@author: shreyanair
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

### Data preprocessing
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_data = dataset_train.iloc[:,1:2].values


'''
# Plot of training data

plt.plot(training_data, color = 'blue', label = 'Real stock price 2017')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Google stock price, 2012 - 2016')
plt.legend()
plt.savefig('Training.png')
plt.show()
'''

sc =  MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(training_data)

# Data structure for 60 time steps and 1 output
# if the number is wrong, it leads to overfitting or wrong predictions

# test with 1,10,20,30,40,50,60

# 60 timesteps =  60 previous financial days (3 prev month)
# X_train -> 60 previous days
# y_train - > next day

X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(X[i-60:i,0])
    y_train.append(X[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (as needed by Keras input)
# here we can add more predictors if they will make an impact on the stock price forecast
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))


### Building the RNN

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


regressor = Sequential()

# adding first layer of LSTM
regressor.add(LSTM(units= 50,return_sequences=True, input_shape = (X_train.shape[1],1)))
    
# Dropuout regularisation
regressor.add(Dropout(0.2))

## NOTE model.summary() can be used to check if same layer has been added

# adding 2nd LSTM layer and dropout regularisation
regressor.add(LSTM(units= 50,return_sequences=True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# adding 3rd LSTM layer and dropout regularisation
regressor.add(LSTM(units= 50,return_sequences=True))
regressor.add(Dropout(0.2))

# adding 4th LSTM layer and dropout regularisation
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))


# Output layer
# We add a classic fully connected layer instead of an LSTM layer
# Dense units param is about number of neurons in this layer, 
# which here will be just 1 (Stock price at time t+1)
regressor.add(Dense(units = 1))

### Compiling RNN
regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')



### Training RNN

regressor.fit(X_train, y_train, epochs= 100, batch_size= 32)




### Forecasting 

# Predicting future stock price
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

# Predicting Stock prices for Jan 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)

# InputShape needed by keras
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)




'''
# Plot for testing data
plt.plot(real_stock_price, color = 'blue', label = 'Real stock price 2017')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Google stock price, Jan 2017')
plt.legend()
plt.savefig('Testing.png')
plt.show()
'''






# Results
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


plt.plot(real_stock_price, color = 'red', label = 'Real stock price 2017')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted stock price 2017')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Trend in Google stock price, Jan 2017')
plt.legend()
plt.savefig('Trend.png')
plt.show()


#from keras.utils import plot_model
#plot_model(regressor, to_file='/Users/shreyanair/Desktop/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/model.png',
#           show_shapes = True, show_layer_names = True)


regressor.summary()

import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
mae = mean_absolute_error(real_stock_price, predicted_stock_price)
coeff_determination = r2_score(real_stock_price, predicted_stock_price)

print("\n RMSE for model : ", rmse)

print("\n Mean Absolute Error for model : ", mae)

print("\n Coefficeint of determination (r^2) for model : ", coeff_determination)











