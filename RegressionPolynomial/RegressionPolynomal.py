#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:56:44 2020

@author: gary
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
            ## Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
## Simple Linear Model
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
print(regressor.score(X_poly, y))

## Result
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X_poly), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()