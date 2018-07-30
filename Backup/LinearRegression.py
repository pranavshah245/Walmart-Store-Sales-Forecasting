# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:40:59 2018

@author: pranav
"""

# Importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

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

# Setting up a dataframe to store Error metrics of all the models
df_ErrorMetrics = pd.DataFrame(columns=['Model', 'R-Square (%)',
                                        'MeanAbsoluteError',
                                        'MeanSquaredError',
                                        'RootMeanSquaredError'])
count = 0

# %%

def linear_regression(X_train, X_test, y_train, y_test, model):
    # Linear Regression Model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    print("Intercept: "+str(lm.intercept_))
    coeff_df = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficient'])
    print(coeff_df)
    predictions = lm.predict(X_test)

    # Error Metrics
    # print('R-Square:', 100*(metrics.r2_score(y_test, predictions)))
    # print('Mean Absolute Error:',
    #       metrics.mean_absolute_error(y_test, predictions))
    # print('Mean Squared Error:',
    #       metrics.mean_squared_error(y_test, predictions))
    # print('Root Mean Squared Error:',
    #       np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    global count
    df_ErrorMetrics.loc[count, 'Model'] = model
    df_ErrorMetrics.loc[count, 'R-Square (%)'] = round(100*(
            metrics.r2_score(y_test, predictions)), 3)
    df_ErrorMetrics.loc[count, 'MeanAbsoluteError'] = round(
            metrics.mean_absolute_error(y_test, predictions), 3)
    df_ErrorMetrics.loc[count, 'MeanSquaredError'] = round(
            metrics.mean_squared_error(y_test, predictions), 3)
    df_ErrorMetrics.loc[count, 'RootMeanSquaredError'] = round(
            np.sqrt(metrics.mean_squared_error(y_test, predictions)), 3)
    count += 1

    # Visualizing the performance of the model
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    sns.set_context('poster')

    sns.jointplot(y_test, predictions, kind='reg', size=8)
    plt.xlabel('Test Set')
    plt.ylabel('Predictions', fontsize=18)
    plt.title('Model Performance', fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.clf()

    sns.distplot((y_test-predictions).dropna(), bins=50)
    plt.xlabel('Residuals')
    plt.ylabel('Counts', fontsize=18)
    plt.title('Histogram of Residuals', fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.clf()

# %%

X = df_CombinedProcessed.drop('Weekly_Sales', axis=1)
y = df_CombinedProcessed['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)

linear_regression(X_train, X_test, y_train, y_test, 'Linear Regression')

# %%

# Scaling all features to a range of [0, 1]
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler1.fit(df_CombinedProcessed)
df_Scaled1 = pd.DataFrame(scaler1.transform(df_CombinedProcessed),
                         columns=df_CombinedProcessed.columns)

X_scaled1 = df_Scaled1.drop('Weekly_Sales', axis=1)
y_scaled1 = df_Scaled1['Weekly_Sales']

X_train_scaled1, X_test_scaled1, y_train_scaled1, y_test_scaled1 = (
        train_test_split(X_scaled1, y_scaled1, test_size=0.3,
                         random_state=101))

linear_regression(X_train_scaled1, X_test_scaled1, y_train_scaled1,
                  y_test_scaled1, 'MinMax Scaled [0,1]')

# %%

# Scaling all features to a range of [-1, 1]
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler2.fit(df_CombinedProcessed)
df_Scaled2 = pd.DataFrame(scaler2.transform(df_CombinedProcessed),
                         columns=df_CombinedProcessed.columns)

X_scaled2 = df_Scaled2.drop('Weekly_Sales', axis=1)
y_scaled2 = df_Scaled2['Weekly_Sales']

X_train_scaled2, X_test_scaled2, y_train_scaled2, y_test_scaled2 = (
        train_test_split(X_scaled2, y_scaled2, test_size=0.3,
                         random_state=101))

linear_regression(X_train_scaled2, X_test_scaled2, y_train_scaled2,
                  y_test_scaled2, 'MinMax Scaled [-1,1]')

# %%

# Normalizing attributes
scaler3 = StandardScaler()
scaler3.fit(df_CombinedProcessed)
df_Scaled3 = pd.DataFrame(scaler3.transform(df_CombinedProcessed),
                          columns=df_CombinedProcessed.columns)

X_scaled3 = df_Scaled3.drop('Weekly_Sales', axis=1)
y_scaled3 = df_Scaled3['Weekly_Sales']

X_train_scaled3, X_test_scaled3, y_train_scaled3, y_test_scaled3 = (
        train_test_split(X_scaled3, y_scaled3, test_size=0.3,
                         random_state=101))

linear_regression(X_train_scaled3, X_test_scaled3, y_train_scaled3,
                  y_test_scaled3, 'Standard Normal Scaled')

# %%

X = df_CombinedProcessed.drop('Weekly_Sales', axis=1)
y = df_CombinedProcessed['Weekly_Sales']

for k in range(1, len(df_CombinedProcessed.columns)):
    X_new = pd.DataFrame(SelectKBest(f_regression, k=k).fit_transform(X, y))
    X_train, X_test, y_train, y_test = train_test_split(X_new, y,
                                                        test_size=0.3,
                                                        random_state=101)
    model_name = str(k)+" Best Features"
    linear_regression(X_train, X_test, y_train, y_test, model_name)

# %%

df_ErrorMetrics.to_csv(path_base+"Regression Results.csv", index=False)

# %%
