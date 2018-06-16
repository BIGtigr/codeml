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
# are in the two dimensions, the two categories of users are going to be
# separated by a straight line, our intuition of logistic regression will be
# better shaped when we find out about the graphical results
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
