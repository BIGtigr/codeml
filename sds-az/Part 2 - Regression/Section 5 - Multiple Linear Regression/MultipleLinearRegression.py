#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Apr 14 18:21:37 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = '50_Startups.csv'
Dependent_Variable_Column = 4
Categorical_Column = 3
Test_Data_Size = 1 / 5          # 80:20 ratio

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, Dependent_Variable_Column].values

# encoding categorical feature values to numeric values
labelEncoder = LabelEncoder()
X[:, Categorical_Column] = labelEncoder.fit_transform(X[:, Categorical_Column])

# creating dummy variables for the encoded categorical features
oneHotEncoder = OneHotEncoder(categorical_features=[Categorical_Column])
X = oneHotEncoder.fit_transform(X).toarray()

# avoid the dummy variable trap, discard one of the dummy variables
X = X[:, 1:]

# splitting the dataset into Training set and the Test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=Test_Data_Size,
    random_state=0
)

# fitting multiple linear regression model to the training data set
linRegressor = LinearRegression()
linRegressor.fit(X_train, y_train)

# testing the performance of our model on test data set by predicting the
# dependent variable for test data set
y_pred = linRegressor.predict(X_test)
