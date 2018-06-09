#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Wed Jun  6 00:33:49 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

"""
# feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X.reshape(-1, 1))
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))
"""

# ============================================================================ #

# creating and fitting the decision tree model to the dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# predicting a new result with decision tree model
# the sample should also be a 1 x m matrix with m feature values
sampleValue = np.array([[6.5]])
y_pred = regressor.predict(sampleValue)

# ============================================================================ #

# visualising the decision tree regression results
# DTR trap ::
# Unlike other previous models, DTR model is not continuous
# The DTR model works by considering the entropy and the information gain and
# splitting the independent feature variables into several intervals. So the
# different intervals form straight line partitions in 1D feature, or some 2D
# rectangles in 2D feature, & higher dimensional boxes in multi-dim feature.
# In each of the boxes we take the average value of the dependent variable and
# predict that average value for all the values of independent variables lying
# in that box.
# So for one feature variable and one dependent variable, we have the intervals
# on x-axis with constant predicted value of y on y-axis in that interval. This
# gives us constant horizontal line in each of the intervals and such lines of
# different intervals connected with each other like steps of a ladder.
# If the line within an interval is not constant (horizontal) then it tells us
# that
# 1. Either the plot has such data due to which it considers infinite number of
#   intervals instead of fixed number of intervals & the infinite constant
#   values for each of those intervals connected with each other, creating a
#   slant line in a wider view.
# 2. Or we have a problem with the data such that the number of intervals are
#   not sufficient and there are no predictions to plot in between the
#   intervals. The actual predictions are directly joined by straight lines &
#   may not be perfectly horizontal or almost vertical.
# Such situation is called DTR trap or red flag.
# In our case first option is not possible as our data has limited x-values.
# So we need to have higher resolution to visualize.
#
# We may use the resolution of actual data to plot the lines only when the
# model is continuous. The DTR model is not continuous, so using the actual
# resolution of data (if lower) may lead to trap situation and the curve may
# not look correct/smooth. This thing is prominent only in non-linear models as
# they are rendered as curves, linear models have straight lines so this thing
# is not visible there. Some examples are:
# continuous linear models => uni & multivariate linear regression
# continuous non-linear models => polynomial, support vector regression 
# non-continuous non-linear models => decision tree regression

stepSize = 0.01
X_grid = np.arange(start=min(X), stop=max(X)+stepSize, step=stepSize)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red', marker='o', label='Samples')
plt.plot(X_grid,
         regressor.predict(X_grid),
         color='blue',
         label='Decision Tree Model')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()

# %%

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus

# export the decision tree as png image
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data, filled=True,
                feature_names=list(dataset.drop(['Position', 'Salary'], 1)),
                rounded=True, special_characters=True, leaves_parallel=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# render the decision tree image in the console
#from IPython.display import Image
#Image(graph.create_png())
