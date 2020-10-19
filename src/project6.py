#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:59:44 2020

@author: gracemcmonagle
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

filepath = '/Users/gracemcmonagle/Desktop/School/Fall 2020/EECS 731/Project 6/data/TravelTime_451.csv'
df = pd.read_csv(filepath)

df['timestamp'] = pd.to_datetime(
                          df['timestamp'], 
                          format='%Y-%m-%d %H:%M:%S')

df = df.sort_values(by = "timestamp")
first_time = df.iloc[0]['timestamp']

#find difference in time between the first obs and all the rest
df['timestamp'] = df['timestamp'].transform(lambda x : x - first_time)

#find the number of minutes past the first observation of all times
df['timestamp'] = pd.to_numeric(df['timestamp'])/60000000000
#%%
plt.plot(df['timestamp'], df['value'])
plt.show()


plt.hist(df['value'])

#%%
clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
y_pred = clf.fit_predict(df)
X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(df['timestamp'], df['value'], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(df['timestamp'], df['value'], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
#plt.xlim((-5, 5))
#plt.ylim((-5, 5))
#plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()



#%% Isolation Forest
clf = IsolationForest(max_samples=100, random_state=42)
outliers = clf.fit_predict(df)

def pltcolor(lst):
    cols = []
    for l in lst:
        if l==1:
            cols.append('black')
        else:
            cols.append('red')
    return cols

cols = pltcolor(outliers)
plt.scatter(df['timestamp'], df['value'],s =2, c=cols)
plt.show()