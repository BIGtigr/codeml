#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Apr  9 22:12:35 2018
@author     : Sourabh
"""

# %%

# import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split    # cross_validation
from sklearn.preprocessing import StandardScaler  # for standardized scaling

# from sklearn.preprocessing import MinMaxScaler    # for normalized scaling

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

# splitting the dataset into Training set and the Test set
# can also be done separately for X and y in two different statements
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# feature scaling: standardization and normalization
# as the dependent variable y is a categorical variable and takes on only a
# few integer values, we need not scale it for classifier
# We would have to scale it if it takes a huge range of values for regression
#
# the following example is of standardization i.e.,
# removing the mean and scaling to unit variance, ( x - µ ) / σ
# feature scaling on X_test is same as on X_train because we fit the
# StandardScaler object on X_train and used the same fitting to transform
# both X_train and X_test
# it is very important to fit the sc_X on X_train first so that both X_train
# and X_test are scaled on the same basis
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#
# or
#
# the following example is of normalization i.e.,
# scaling each feature on the basis of its range, ( x - min ) / ( max - min )
# norm_X = MinMaxScaler(feature_range=(-1, 1))
# X_train = norm_X.fit_transform(X_train)
# X_test = norm_X.transform(X_test)
#
print(X_train)
print(X_test)
