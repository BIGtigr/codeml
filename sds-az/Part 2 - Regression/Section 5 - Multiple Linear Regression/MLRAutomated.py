#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Wed May 16 20:20:03 2018
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#==============================================================================#

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
    values=np.ones((X.shape[0],), dtype=np.int),
    axis=1
)

#==============================================================================#

# Backward Elimination consists of including all independent variables at once
# and then removing one by one those that are not statistically significant.
# For this model we are wondering what the probability of two variables being
# related is, that is our null hypothesis.

# Step 1: Select a Significance Level
SL = 0.05

# Step 2.1: Initially include all possible regressors (features) in the optimal
# data set.
X_optimal = X[:, :]

# Repeating steps 3, 4 and 5 until our model is finalized
numFeatures = len(X_optimal[0])
for i in range(0, numFeatures):
    # Step 2.2: Fit the new regressor model to the optimal data set
    regressor_OLS = sm.OLS(endog=y, exog=X_optimal).fit()
    # Step 3: Find the maximum P-value and check if it is greater than the SL
    maxPvalue = max(regressor_OLS.pvalues).astype(float)
    print('P =', maxPvalue,
          '\tR2 =', regressor_OLS.rsquared,
          '\tR2A =', regressor_OLS.rsquared_adj)
    if maxPvalue > SL:
        for j in range(0, numFeatures - i):
            if regressor_OLS.pvalues[j].astype(float) == maxPvalue:
                # Step 4: Remove the variable as is is correlated with y
                X_optimal = np.delete(X_optimal, j, 1)
    else:
        # Step 5: our model is ready
        break

# here the marketing spend feature generated a P-value of 0.06 which is just
# slightly above the significance level to stay in the model. So instead of
# directly removing it from the model as we did above, we can also use other
# powerful metrics like R-squared and Adjusted R-squared to decide with more
# certainty whether we need to keep Marketing Spend or remove it. But as far
# as Backward Elimination is concerned, we have removed it and our model is
# ready. So we stop the regression here.
regressor_OLS.summary()

# splitting the optimal dataset into Training set and the Test set
X_train, X_test, y_train, y_test = train_test_split(
    X_optimal,                  # finalised using backward elimination algo
    y,
    test_size=Test_Data_Size,
    random_state=0
)

#==============================================================================#

# fitting optimal multiple linear regression model to the training data set
linRegressor = LinearRegression()
linRegressor.fit(X_train, y_train)

# testing the performance of our model on test data set by predicting the
# dependent variable for test data set
y_pred = linRegressor.predict(X_test)

#==============================================================================#

# used to plot the model
y_train_pred = linRegressor.predict(X_train)

# plot the model and predictions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
x_axis_train = X_train[:, 0]
y_axis_train = X_train[:, 1]
z_axis_train = y_train
#
ax.scatter(x_axis_train, y_axis_train, z_axis_train, c='r', marker='o')
#
x_axis_test = X_test[:, 0]
y_axis_test = X_test[:, 1]
z_axis_test = y_test
#
ax.scatter(x_axis_test, y_axis_test, z_axis_test, c='g', marker='o')
#
x_axis_model = X_train[:, 0]
y_axis_model = X_train[:, 1]
z_axis_model = y_train_pred
#
ax.plot(x_axis_model, y_axis_model, z_axis_model)
#
ax.set_xlabel('Intercept')
ax.set_ylabel('R&D Spend')
ax.set_zlabel('Profit')
#
plt.show()

#==============================================================================#

# Comparison of Adjusted-R2 with R2
#
# predictors: [0, 1, 2, 3, 4, 5]
# Max P-value:                     0.989794124161
# R-squared:                       0.950752484336
# Adj. R-squared:                  0.945156175737
#
# predictors: [0, 1, 3, 4, 5]
# Max P-value:                     0.939832977258
# R-squared:                       0.950752299106
# Adj. R-squared:                  0.946374725693
#
# predictors: [0, 3, 4, 5]
# Max P-value:                     0.60175510785
# R-squared:                       0.950745994068
# Adj. R-squared:                  0.94753377629
#
# predictors: [0, 3, 5]
# Max P-value:                     0.0600303971911
# R-squared:                       0.950450301556
# Adj. R-squared:                  0.94834180375
#
# predictors: [0, 3]
# Max P-value:                     2.78269692297e-24
# R-squared:                       0.94653531608
# Adj. R-squared:                  0.945421468499
#
# R2:
# Every predictor added to a model always increases R-squared.
# Every predictor removed from a model always decreases R-squared.
# This happens irrespective of the significance of the independent variable.
# So it can not be used to verify the goodness of our model.
#
# Adj. R2 or R2A:
# Every predictor added to a model increases R2A if it is significant for y
# otherwise it decreases the R2A.
# Every predictor removed from a model decreases R2A if it is significant for y
# otherwise it increases R2A.
# So it is better to use R2A instead of R2 to verify the goodness of our model.
#
# As the R2A value decreases on removing the variable 'Marketing Spend' which
# is at column 5 of original input X, so we may also decide to keep it in the
# model. This is different from if we use P-value to decide because the P-value
# of Marketing Spend is greater than the 5% significance level.
