#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:57:39 2020

@author: gary
"""


"""
Created on Thu Jul  9 14:03:48 2020

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

            ## Import Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values


                    ## Split the dataset
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


                    ## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

                    ## Model
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel='rbf')
classifier.fit(X_train,y_train)
print(classifier.score(X_test,y_test))

            ## Predict
y_pred = classifier.predict(X_test)

        ## Confusion matrice
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualiser les résultats
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Test set Result | score 93%')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


