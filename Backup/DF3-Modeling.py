# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:58:51 2018

@author: pranav
"""

# Importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import datetime
import time

# Path where the processed data is stored
path_base = "E:/Kaggle/DemandForecasting/Walmart/2ProcessedData/"
output_base = "E:/Kaggle/DemandForecasting/Walmart/3Analysis_Pranav/"

# Filename of the processed data
filename_CombinedData = "combinedprocessed.csv"

# %%

# Reading the processed data for modeling purpose

df_CombinedData = pd.read_csv(path_base + filename_CombinedData,
                              index_col='Date')

df_CombinedData.drop(['Unnamed: 0', 'IsHoliday_y'], axis=1, inplace=True)
df_CombinedData.head(3)
df_CombinedData.info()

# %%

# Pre=processing the data

df_CombinedProcessed = df_CombinedData.copy()
df_CombinedProcessed.dropna(axis=0, inplace=True)

df_CombinedProcessed['Holiday'] = pd.get_dummies(
        df_CombinedProcessed['IsHoliday_x'], drop_first=True)

df_CombinedProcessed[['IsTypeB','IsTypeC']] = pd.get_dummies(
        df_CombinedProcessed['Type'], drop_first=True)

df_CombinedProcessed.drop(['IsHoliday_x', 'Type'], axis=1, inplace=True)
df_CombinedProcessed.head(3)
df_CombinedProcessed.info()

# %%

# Function to frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# %%

# Function to create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# %%

# Function to invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# %%

# Function to scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# %%

# Function to inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# %%

# Funciton to Train RNN LSTM
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1],
                                               X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0,
                  shuffle=False)
        model.reset_states()
    return model

# %%

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# %%

sales = pd.Series(df_CombinedData['Weekly_Sales'])

# %%

# transform data to be stationary
raw_values = sales.values
diff_values = difference(raw_values, 1)

# %%

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# %%

# split data into train and test-sets
train, test = train_test_split(supervised_values, test_size=0.3,
                               random_state=101)

# %%

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# %%

# Hyperparameters
BatchSize = [2, 3, 137, 359]
Epochs = [50, 100, 150, 200]
Neurons = [4, 6, 10, 12]

# Setting up a dataframe to store Error metrics of all the models
df_ErrorMetrics = pd.DataFrame(columns=['BatchSize', 'NumberOfEpochs',
                                        'Neurons', 'RMSE'])
count = 0

# %%

for i in BatchSize:
    for j in Epochs:
        for k in Neurons:
            # Timing the code performance for training
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "Start of LSTM Training for "+str(i)+" batch size, "+
                  str(j)+" epochs and "+str(k)+" Neurons")
            StartTrainTime = time.time()

            # Fit the model
            lstm_model = fit_lstm(train_scaled, i, j, k)

            EndTrainTime = time.time()
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "End of LSTM Training for "+str(i)+" batch size, "+
                  str(j)+" epochs and "+str(k)+" Neurons")
            TotalTrainTime = round((EndTrainTime - StartTrainTime)/60, 2)

            print("Total Time to Train the Data: "+str(TotalTrainTime)+
                  " minutes.")

            # forecast the entire training dataset to build up state for
            # forecasting
            train_reshaped = train_scaled[:, 0].reshape(len(
                    train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=1)

            # walk-forward validation on the test data
            predictions = list()

            # Timing the code performance for validation
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "Start of Validation for "+str(i)+" batch size, "+
                  str(j)+" epochs and "+str(k)+" Neurons")
            StartValidateTime = time.time()

            for l in range(len(test_scaled)):
                # make one-step forecast
                X, y = test_scaled[l, 0:-1], test_scaled[l, -1]
                yhat = forecast_lstm(lstm_model, 1, X)
                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # invert differencing
                yhat = inverse_difference(raw_values, yhat,
                                          len(test_scaled)+1-l)
                # store forecast
                predictions.append(yhat)
                expected = raw_values[len(train) + l + 1]
                # print('Predicted=%f, Expected=%f' % (yhat, expected))

            # split raw data in train and test set
            raw_train, raw_test = train_test_split(raw_values, test_size=0.3,
                                                   random_state=101)

            # report performance
            rmse = np.sqrt(metrics.mean_squared_error(raw_test, predictions))
            print("RMSE for "+str(i)+" batch size, "+ str(j)+" epochs and "+
                  str(k)+" Neurons: %.3f" % rmse)

            EndValidateTime = time.time()
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  "End of Validation for "+str(i)+" batch size, "+
                   str(j)+" epochs and "+str(k)+" Neurons")
            TotalValidateTime = round(
                    (EndValidateTime - StartValidateTime)/60, 2)
            print("Total Time to Validate the Model: "+str(TotalValidateTime)+
                  " minutes.")

            global count
            df_ErrorMetrics.loc[count, 'BatchSize'] = int(i)
            df_ErrorMetrics.loc[count, 'NumberOfEpochs'] = int(j)
            df_ErrorMetrics.loc[count, 'Neurons'] = int(k)
            df_ErrorMetrics.loc[count, 'RMSE'] = round(float(rmse),3)
            count += 1

            # line plot of observed vs predicted
            plt.figure(figsize=(12, 8))
            sns.set_style('whitegrid')
            sns.set_context('poster')

            plt.plot(raw_test[-500:], lw=1.5)
            plt.plot(predictions[-500:], lw=1.5)
            plt.legend()
            plt.xlabel('Records')
            plt.ylabel('Weekly Sales', fontsize=18)
            plt.title("Observed vs Predicted for "+str(i)+" batch size, "+
                      str(j)+" epochs and "+str(k)+" Neurons")
            plt.title('', fontsize=18)
            plt.tight_layout()
            plt.savefig("Observed vs Predicted for "+str(i)+" batch size, "+
                        str(j)+" epochs and "+str(k)+" Neurons.png")
            plt.show()
            plt.clf()

# %%

df_ErrorMetrics.to_csv(path_base+"LSTM Results.csv", index=False)

# %%