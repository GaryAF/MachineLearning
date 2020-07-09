#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:04:39 2020

@author: gary
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

            ## Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

                    ## Split the dataset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 1.0/3, random_state = 0)

## No need to Feature Scalling because of linear regression

from sklearn.linear_model import LinearRegression

## Simple Linear Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))

## Result
plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()