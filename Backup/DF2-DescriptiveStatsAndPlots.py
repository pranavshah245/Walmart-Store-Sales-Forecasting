# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:20:22 2018

@author: pranav
"""

# Importing the necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssaplot as ssa

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

df_CombinedProcessed = df_CombinedData.copy()

df_CombinedProcessed.dropna(axis=0, inplace=True)

# %%

# Visualizing the data
plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.pairplot(df_CombinedProcessed, hue='IsHoliday_x', palette='rainbow')
plt.title('Pairwise Comparison', fontsize=18)
plt.tight_layout()
plt.savefig('Pairwise_Comparison.png')
plt.show()
plt.clf()

# %%


g = sns.PairGrid(df_CombinedProcessed)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.title('Pairwise Comparison_2', fontsize=18)
plt.tight_layout()
plt.savefig('Pairwise_Comparison_2.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.heatmap(df_CombinedProcessed.corr(), cmap='coolwarm')
plt.title('Heatmap of Pearson Coorelations', fontsize=18)
plt.tight_layout()
plt.savefig('Heatmap of Pearson Coorelations.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster', font_scale=1)

ax = sns.countplot(df_CombinedProcessed['Type'], palette='coolwarm')
sns.despine()
ssa.annotate(ax)
ssa.annotate(ax, location='Top', message='Count')
ssa.annotate(ax, location='Middle', message='Percentage')
plt.xlabel('Type', fontsize=18)
plt.ylabel('Number of Records', fontsize=18)
plt.title('Number of Records for each Type', fontsize=18)
plt.tight_layout()
plt.savefig('Number of Records for each Type.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster', font_scale=1)

ax = sns.countplot(df_CombinedProcessed['IsHoliday_x'], palette='RdBu_r')
sns.despine()
ssa.annotate(ax)
ssa.annotate(ax, location='Top', message='Count')
ssa.annotate(ax, location='Middle', message='Percentage')
plt.xlabel('Holiday (True/False)', fontsize=18)
plt.ylabel('Number of Records', fontsize=18)
plt.title('Number of Holidays', fontsize=18)
plt.tight_layout()
plt.savefig('Number of Holidays.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.distplot(df_CombinedProcessed['Weekly_Sales'].dropna(), kde=False)
plt.yscale('log')
plt.xlabel('Weekly_Sales', fontsize=18)
plt.ylabel('Fraction of Counts', fontsize=18)
plt.title('Histogram of Weekly Sales', fontsize=18)
plt.tight_layout()
plt.savefig('Histogram of Weekly Sales.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.distplot(df_CombinedProcessed['Temperature'].dropna(), kde=False)
# plt.yscale('log')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Fraction of Counts', fontsize=18)
plt.title('Histogram of Temperature', fontsize=18)
plt.tight_layout()
plt.savefig('Histogram of Temperature.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.distplot(df_CombinedProcessed['Fuel_Price'].dropna(), kde=False)
# plt.yscale('log')
plt.xlabel('Fuel_Price', fontsize=18)
plt.ylabel('Fraction of Counts', fontsize=18)
plt.title('Histogram of Fuel Price', fontsize=18)
plt.tight_layout()
plt.savefig('Histogram of Fuel Price.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.distplot(df_CombinedProcessed['CPI'].dropna(), kde=False)
# plt.yscale('log')
plt.xlabel('CPI', fontsize=18)
plt.ylabel('Fraction of Counts', fontsize=18)
plt.title('Histogram of CPI', fontsize=18)
plt.tight_layout()
plt.savefig('Histogram of CPI.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.distplot(df_CombinedProcessed['Unemployment'].dropna(), kde=False)
# plt.yscale('log')
plt.xlabel('Unemployment', fontsize=18)
plt.ylabel('Fraction of Counts', fontsize=18)
plt.title('Histogram of Unemployment', fontsize=18)
plt.tight_layout()
plt.savefig('Histogram of Unemployment.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.violinplot(x='Type', y='Weekly_Sales', data=df_CombinedProcessed,
               hue='IsHoliday_x')
# plt.yscale('log')
plt.xlabel('Type', fontsize=18)
plt.ylabel('Weekly_Sales', fontsize=18)
plt.title('Violin Plot of Type vs Weekly_Sales', fontsize=18)
plt.tight_layout()
plt.savefig('Violin Plot of Type vs Weekly_Sales.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(15, 6))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.lmplot(x='Temperature', y='Weekly_Sales', data=df_CombinedProcessed,
           col='IsHoliday_x', size=8, aspect=0.6, palette='coolwarm')
# plt.yscale('log')
plt.xlabel('Temperature', fontsize=18)
plt.ylabel('Weekly_Sales', fontsize=18)
# plt.title('Regression Plot of Temperature vs Weekly_Sales', fontsize=18)
plt.tight_layout()
plt.savefig('Regression Plot of Temperature vs Weekly_Sales.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('darkgrid')
sns.set_context('poster')

sns.jointplot(x='Unemployment', y='Weekly_Sales', data = df_CombinedProcessed,
              size=8, kind='reg')
# plt.yscale('log')
plt.xlabel('Unemployment', fontsize=18)
plt.ylabel('Weekly_Sales', fontsize=18)
plt.title('Regression Plot of Unemployment vs Weekly_Sales', fontsize=18)
plt.tight_layout()
plt.savefig('Regression Plot of Unemployment vs Weekly_Sales.png')
plt.show()
plt.clf()


# %%

plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
sns.set_context('poster')

g = sns.FacetGrid(data=df_CombinedProcessed, col='IsHoliday_x', size=8,
                  palette='coolwarm')
g.map(plt.hist, 'Weekly_Sales')
# plt.yscale('log')
plt.xlabel('Weekly_Sales', fontsize=14)
plt.ylabel('Fraction of Counts', fontsize=14)
 #plt.title('Histogram of Weekly Sales per Holiday', fontsize=14)
plt.tight_layout()
plt.savefig('Histogram of Weekly Sales per Holiday.png')
plt.show()
plt.clf()

# %%

plt.figure(figsize=(12, 8))
sns.set_style('whitegrid')
sns.set_context('poster')

g = sns.FacetGrid(data=df_CombinedProcessed, col='Type', size=8,
                  palette='coolwarm')
g.map(plt.hist, 'Weekly_Sales')
# plt.yscale('log')
plt.xlabel('Weekly_Sales', fontsize=14)
plt.ylabel('Fraction of Counts', fontsize=14)
# plt.title('Histogram of Weekly Sales per Type', fontsize=14)
plt.tight_layout()
plt.savefig('Histogram of Weekly Sales per Type.png')
plt.show()
plt.clf()

# %%
