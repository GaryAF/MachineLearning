#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:06:05 2020

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

            ## Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)

                        ## Category variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = ColumnTransformer([('my_ohe', OneHotEncoder(), [3])], remainder='passthrough')
X = onehotencoder.fit_transform(X)
Xa= X[:,1:]

 ## SPlit data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Model Linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))

