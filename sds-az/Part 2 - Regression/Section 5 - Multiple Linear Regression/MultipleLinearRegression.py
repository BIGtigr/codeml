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
import statsmodels.formula.api as sm

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
# this is so important step that the library takes care of it automatically
# so we don't need to do it here manually
X = X[:, 1:]

# splitting the dataset into Training set and the Test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=Test_Data_Size,
    random_state=0
)

# ============================================================================ #

# fitting multiple linear regression model to the training data set
linRegressor = LinearRegression()
linRegressor.fit(X_train, y_train)

# testing the performance of our model on test data set by predicting the
# dependent variable for test data set
y_pred = linRegressor.predict(X_test)

y_diff = y_test - y_pred

# ============================================================================ #

# building the optimal model using Backward Elimination
# we need to see that the equation for multiple linear regression is:
# y = b0 + b1*x1 + b2*x2 + ... + bn*xn
# which can be treated as
# y = b0*x0 + b1*x1 + b2*x2 + ... + bn*xn
# where x0 is 1 for all samples in X
# most of the libraries are aware of this fact and take care of this
# but the statsmodel library does not take care of this automatically
# so we can insert an additional column in our matrix X to represent x0
# all the values in X[:, 0] will be 1s
X = np.insert(
    arr=X,
    obj=0,      # index of the column where want to insert new column
    values=np.ones((X.shape[0],), dtype=np.float64),
    axis=1
)
# or
#X = np.append(
#        arr=np.ones((X.shape[0],1)),
#        values=X,
#        axis=1
#        )

# Backward Elimination consists of including all independent variables at once
# and then removing one by one those that are not statistically significant.
# For this model we are wondering what the probability of two variables being
# related is, that is our null hypothesis.

# Step 1: Select a Significance Level
SL = 0.05

# Step 2.1: Initially include all possible regressors (features) in the optimal
# data set. Therefore, all column indices have been mentioned explicitly here.
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]

# Step 2.2: Fit the new regressor model to the optimal data set
regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()

# Step 3: Select the predictor with highest P-value
regressor_OLS.summary()
