#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
Created on  : Mon Apr  9 22:12:35 2018
@author     : Sourabh
"""

#%%

# import the libraries
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.nan)

# import the dataset
dataset = pd.read_csv('Data.csv')
print(dataset)
print(dataset.values)

# extract the feature and the dependent variable vectors
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, 3].values
print(y)
