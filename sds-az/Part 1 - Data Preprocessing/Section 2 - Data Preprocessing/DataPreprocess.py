#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Apr  9 22:12:35 2018
@author     : Sourabh
"""

#%%

# import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

np.set_printoptions(threshold=np.nan)

# import the dataset
dataset = pd.read_csv('Data.csv')
print(dataset)
print(dataset.values)

# extract the feature and the dependent variable vectors
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, 3].values
print(y)

# taking care of the missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# encoding categorical data
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
print(X)
y = labelencoder.fit_transform(y)
print(y)
