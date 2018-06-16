#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Fri Jun 15 20:30:21 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'Social_Network_Ads.csv'
Test_Data_Size = 0.25

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# splitting the dataset into Training set and the Test set
# can also be done separately for X and y in two different statements
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Test_Data_Size,
        random_state=0
        )

# features need to be scaled so as to match range of values with each other
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# logistic regression is a linear classifier, which means that since here we
# have two independent variables so we are in the two dimensions, the two
# categories of users are going to be separated by a straight line, our
# intuition of logistic regression will be better shaped when we find out about
# the graphical results
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predicting the test set results
y_pred = classifier.predict(X_test)

# evaluating if our logistic regression model learnt and understood the
# correlations in the training data set correctly to see if it can make
# powerful predictions on new data set like test data set
# we will make use of the confusion matrix, which contains the correct
# predictions that our model made on the test data set as well as the incorrect
# predictions
# we compute confusion matrix to evaluate the accuracy of a classification
# a confusion matrix C is such that C[i][j] is equal to the number of
# observations known to be in group i but predicted to be in group j.
# thus in binary classification,
# true negatives is C[0][0] => from false to false is true
# false positives is C[0][1] => from false to true is false
# false negatives is C[1][0] => from true to false is false
# true positives is C[1][1] => from true to true is true
# true negatives and true positives are considered as correct predictions
# false negatives and false positives are considered as wrong predictions
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tn + tp) / y_test.shape[0] * 100
print('Accuracy = %.2f%%' % accuracy)

# visualizing the predictions
# this graph represents all social network users from our training set
# plotted according to their age against their salary represented in the form
# of points
# some points are represented with red colour and some with green colour which
# represents the observation for all users of the training set i.e., if they
# bought the SUV (y=1) then the colour is green and if not (y=0) then the
# colour is red
# the graph has a straight line which is called the decision boundary and is
# drawn to represent the classifier model
# on one side there is a red region to represent the group not buying the SUV
# on other side there is a green region to show the group buying the SUV
# analysis:
# the users who are young with low salary did not buy the SUV
# the aged users with high salary bought the SUV as it is a big car for family
# some old people with low salary also bought the SUV due to many reasons
# some young users with high salary also bought the SUV doe to many reasons
# goal:
# the goal of classification is to classify the users in the correct categories
# i.e., the users represented as green dots should be classified in the green
# region and the users drawn as red dots should be classified in the red region.
# For prediction, if a user lies in the green prediction region, the model will
# predict that he will buy the SUV and if a user lies in the red prediction
# region, the classifier will predict him not buying the SUV.
# So the point is the grounded truth which happened in reality and the region
# is the prediction.
# important:
# the decision boundary is a straight line because the logistic regression
# classifier is a linear classifier and we are in a 2D space due to 2 feature
# variables. If we had 3 feature variables, we would have been in a 3D space,
# and the decision boundary would have been a 3D plane. It will always be a
# straight line or plane or space if the classifier is linear
# decision boundary is not straight if the classifier is not linear.
# prediction:
# correct prediction is red dots in red region and green dots in green region
# wrong prediction is red dots in green region and green dots in red region
# the incorrect predictions are specifically due to the fact that our data set
# is not linearly distributed but the classifier is a linear classifier
def plot2D(X, y, title, labels):
    X1_min = X[:, 0].min() - 1
    X1_max = X[:, 0].max() + 1
    X2_min = X[:, 1].min() - 1
    X2_max = X[:, 1].max() + 1
    X1, X2 = np.meshgrid(
            np.arange(start = X1_min, stop = X1_max, step = 0.01),
            np.arange(start = X2_min, stop = X2_max, step = 0.01)
            )
    generated_points = np.array([X1.ravel(), X2.ravel()]).T
    plt.contourf(
            X1,
            X2,
            classifier.predict(generated_points).reshape(X1.shape),
            alpha = 0.5,
            cmap = ListedColormap(('red', 'green'))
            )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for index, status in enumerate(np.unique(y)):
        plt.scatter(
                X[y == status, 0],
                X[y == status, 1],
                c = ListedColormap(('red', 'green'))(index),
                label = status
                )
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()

plt.figure()

plt.subplot(121)
plot2D(
       X_train,
       y_train,
       title = 'Logistic Regression (Training Set)',
       labels = ['Age', 'Estimated Salary']
       )

plt.subplot(122)
plot2D(
       X_test,
       y_test,
       title = 'Logistic Regression (Test Set)',
       labels = ['Age', 'Estimated Salary']
       )

plt.subplots_adjust(top=0.93, bottom=0.10, left=0.05, right=0.97)
plt.show()
