#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Tue Jun 26 22:55:14 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.set_printoptions(threshold=np.nan)

# ---------------------------------------------------------------------------- #
# constant properties that need changes according to the actual problem
Data_File = 'Social_Network_Ads.csv'
Test_Data_Size = 0.25
Feature_Column_Start = 2
Feature_Column_End = 3
Observation_Column = 4
Plot_Resolution = 0.01
Plot_Category_Transparency = 0.5

# ---------------------------------------------------------------------------- #
# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, [Feature_Column_Start, Feature_Column_End]].values
y = dataset.iloc[:, Observation_Column].values

# ---------------------------------------------------------------------------- #
# splitting the dataset into Training set and the Test set
# can also be done separately for X and y in two different statements
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Test_Data_Size,
        random_state=0
        )

# ---------------------------------------------------------------------------- #
# features need to be scaled so as to match range of values with each other
# use it if not automatically provided by the model
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ---------------------------------------------------------------------------- #
# fitting the classfier to the training data set
classifier = KNeighborsClassifier(
        n_neighbors = 5,
        metric = 'minkowski',    # to be used with power p
        p = 2                    # p = 1 for manhattan, p = 2 for euclidean
        )
classifier.fit(X_train, y_train)

# ---------------------------------------------------------------------------- #
# predicting the test set results
y_pred = classifier.predict(X_test)

# ---------------------------------------------------------------------------- #
# evaluating the results by making a confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tn + tp) / y_test.shape[0] * 100
print('Accuracy = %.2f%%' % accuracy)

# ---------------------------------------------------------------------------- #
# visualizing the predictions - only applicable for a 2D feature space
def plot2D(X, y, title, labels):
    X1_min = X[:, 0].min() - 1
    X1_max = X[:, 0].max() + 1
    X2_min = X[:, 1].min() - 1
    X2_max = X[:, 1].max() + 1
    X1, X2 = np.meshgrid(
            np.arange(start = X1_min, stop = X1_max, step = Plot_Resolution),
            np.arange(start = X2_min, stop = X2_max, step = Plot_Resolution)
            )
    generated_points = np.array([X1.ravel(), X2.ravel()]).T
    plt.contourf(
            X1,
            X2,
            classifier.predict(generated_points).reshape(X1.shape),
            alpha = Plot_Category_Transparency,
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
       title = 'K-NN Classifier (Training Set)',
       labels = ['Age', 'Estimated Salary']
       )

plt.subplot(122)
plot2D(
       X_test,
       y_test,
       title = 'K-NN Classifier (Test Set)',
       labels = ['Age', 'Estimated Salary']
       )

plt.subplots_adjust(top=0.93, bottom=0.10, left=0.05, right=0.97)
plt.show()
