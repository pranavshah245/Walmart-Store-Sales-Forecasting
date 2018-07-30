# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

# %% Offline Mode

from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
init_notebook_mode()

# %% Setup path names
path_base = "E:/SSAGeneralCode/SourceCode/Kaggle/Walmart/1RawDataDoNotEdit/"
ouptut_base = "E:/SSAGeneralCode/SourceCode/Kaggle/Walmart/2ProcessedData"
filename_Store = "stores.csv"
filename_Features = "features.csv"
filename_Train = "train.csv"
filename_merged = "MasterDataFrame.csv"
# %% Load Data
df_Store = pd.read_csv(path_base + filename_Store)
df_Train = pd.read_csv(path_base + filename_Train)
df_Features = pd.read_csv(path_base + filename_Features)

df_Store.shape
df_Train.shape
df_Features.shape

df_Store.head(3)
df_Train.head(3)
df_Features.head(3)

# %% Merge DataSet
df_temp = pd.merge(df_Train, df_Store, how="left", on='Store')
df_combined = pd.merge(df_temp, df_Features, how="left", on=['Store','Date']) 
df_combined.columns

# %% Merged Dataset to CSV
df_combined.to_csv(ouptut_base + filename_merged)
df_combined.describe(include='all')


# %% Cleaning

# change type of categorical vairables to strings
df_combined['Store'] = df_combined['Store'].astype(str)
df_combined['Dept'] = df_combined['Dept'].astype(str)

# change date to datetme
#df_combined['Date'] = pd.to_datetime(df_combined['Date'])
# %% Descriptive Stats

# Number of Types of Stores - Three types of stores
df_combined['Type'].unique()

# Number of Department by Stores - Three types of stores
df_combined[['Store', 'Dept']].groupby(['Store'])['Dept'].nunique()



# %% Histograms

data = [go.Histogram(x=df_combined['Weekly_Sales'])]
plot(data)

# %% Time Series
df_combined.columns
df_combined.index
df_combined.set_index('Date', inplace=True)

#indexer = df_combined.index<= "2012-01-01"
#df_combined = df_combined.loc[indexer,]
data = [go.Scatter(x=df_combined.index, y=df_combined['Weekly_Sales'])]

layout = dict(
    title = "Manually Set Date Range"
    #xaxis = dict(
      #  range = ['2011-07-01','2012-12-31']
 #)
)

fig = go.Figure(data=data)
plot(fig)

