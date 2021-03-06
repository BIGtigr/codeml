#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Apr 11 23:59:59 2018
@author     : Sourabh
"""

# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.nan)

# constant properties that need changes according to the actual problem
Data_File = 'Data.csv'
Dependent_Variable_Column = 3
Test_Data_Size = 0.2

# import the dataset & extract the feature and the dependent variable vectors
dataset = pd.read_csv(Data_File)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, Dependent_Variable_Column].values

# splitting the dataset into Training set and the Test set
# can also be done separately for X and y in two different statements
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=Test_Data_Size,
                                                    random_state=0)

"""
# feature scaling: to be used when needed
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
