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
from sklearn.preprocessing import OneHotEncoder

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
# the values of these features are categorical and have no relational
# ordering with respect to various possible encoded integer values
# to resolve this we use OneHotEncoder and introduce as many new columns as
# there are categorical values of a feature
# each category of the feature is kept as 1s in its respective column, rest 0s
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
# the ordering relationship of categorical values of dependent variable does
# not happen because the model knows that the dependent variable would have
# categorical values, so no need to use OneHotEncoder for it
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
