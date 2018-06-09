#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Sat Jun  9 17:13:05 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ============================================================================ #

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'Position_Salaries.csv'
Dependent_Variable_Column = 2

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, 1:Dependent_Variable_Column].values
y = dataset.iloc[:, Dependent_Variable_Column].values

# ============================================================================ #

# creating and fitting the random forest regression model to the dataset
# as the number of trees are increased, the better is the convergence of the
# average value of each of the trees. The number of steps may also increase
# with the number of intervals but not equally, the number of steps actually
# depend on the entropy and information gain and may not increase at all.
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# predicting a new result with random forest regression model
# the sample should also be a 1 x m matrix with m feature values
sampleValue = np.array([[6.5]])
y_pred = regressor.predict(sampleValue)

# ============================================================================ #

# visualizing non-continuous model prediction using higher resolution
# as the stepSize is further reduced, the steps become more and more vertical
stepSize = 0.01
X_grid = np.arange(start=min(X), stop=max(X)+stepSize, step=stepSize)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', marker='o', label='Samples')
plt.plot(X_grid,
         regressor.predict(X_grid),
         color='blue',
         label='RFR Model')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()
