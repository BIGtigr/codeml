#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Wed Apr 11 23:21:00 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# %%

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'Salary_Data.csv'
Dependent_Variable_Column = 1
Test_Data_Size = 1/3

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, Dependent_Variable_Column].values

# splitting the dataset into Training set and the Test set
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Test_Data_Size,
        random_state=0
        )

"""
# feature scaling: to be used when needed
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# %%

# fitting simple linear regression to the training set
linRegressor = LinearRegression()
linRegressor.fit(X_train, y_train)

# predicting the test results
y_pred = linRegressor.predict(X_test)

# %%

y_train_pred = linRegressor.predict(X_train)

# visualising the training set results
# scatter plot plots the each of the data points individually
plt.scatter(X_train, y_train, c='red', marker='x', label='Training Data')
# to plot the regression line for the actual salary cases we can use line plot
# the x-coordinate will be the actual training data set
# the y-coordinate will be the predicted vector on the actual training data set
# we can not use y_pred as y-coordinate as it is based on X_test not X_train
plt.plot(X_train, y_train_pred, color='blue', label='Model')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc='upper-left')
plt.show()
