#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat May 19 16:40:33 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================================ #

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'name.csv'
Dependent_Variable_Column = 2
Test_Data_Size = 0.2

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, :Dependent_Variable_Column].values
y = dataset.iloc[:, Dependent_Variable_Column].values

"""
# splitting the dataset into Training set and the Test set
# can also be done separately for X and y in two different statements
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Test_Data_Size,
        random_state=0
        )
"""

"""
# feature scaling: to be used whenever needed
# most of the regression libraries do this automatically
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# ============================================================================ #

# creating and fitting the regression model to the dataset
# TODO: need to write code here
regressor = None

# predicting a new result with regression model
# TODO: the sample should also be a 1 x m matrix with m feature values
sampleValue = []
y_pred = regressor.predict(sampleValue)

# ============================================================================ #

# visualising the regression results
plt.scatter(X, y, color='red', marker='o', label='Samples')
plt.plot(X,
         regressor.predict(X),
         color='blue',
         label='Regression Model')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()

"""
# visualising the regression results (for smoother curve)
stepSize = 0.1
X_grid = np.arange(start=min(X), stop=max(X)+stepSize, step=stepSize)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', marker='o', label='Samples')
plt.plot(X_grid,
         regressor.predict(X_grid),
         color='blue',
         label='Regression Model')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()
"""
