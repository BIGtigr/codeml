#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Jun  4 23:17:56 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# ============================================================================ #

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'Position_Salaries.csv'
Dependent_Variable_Column = 2
Test_Data_Size = 0.2

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, 1:Dependent_Variable_Column].values
y = dataset.iloc[:, Dependent_Variable_Column].values

# feature scaling: SVR does not support it automatically, we need to do it here
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X.reshape(-1, 1))
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

# ============================================================================ #

# creating and fitting the SVR model to the dataset
# as we know that our training data set is not linear, we should not use linear
# kernel here, it's better we use any of Polynomial or Gaussian kernel.
regressor = SVR(kernel='rbf')
regressor.fit(X_scaled, y_scaled)

# predicting a new result with SVR model
# the sample should also be a 1 x m matrix with m feature values
sampleValue = np.array([[6.5]])
y_pred = sc_y.inverse_transform(
            regressor.predict(
                sc_X.transform(sampleValue)
                )
            )

# ============================================================================ #

# visualising the SVR results
stepSize = 0.1
X_grid = np.arange(start=min(X), stop=max(X)+stepSize, step=stepSize)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', marker='o', label='Samples')
plt.plot(X_grid,
         sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))),
         color='blue',
         label='SVR Model')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()
