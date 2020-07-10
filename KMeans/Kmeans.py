#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:03:48 2020

@author: gary
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

            ## Import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

## Elbow method to find optimal nb_clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0 , init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Elbow method")
plt.xlabel('Nb Clusters')
plt.ylabel("WCSS")
plt.show()

# Model
kmeans = KMeans(n_clusters = 5, random_state = 0 , init='k-means++')
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 1 , 0], X[y_kmeans == 1 , 1], c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2 , 0], X[y_kmeans == 2 , 1], c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3 , 0], X[y_kmeans == 3 , 1], c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 4 , 0], X[y_kmeans == 4 , 1], c = 'magenta', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 0 , 0], X[y_kmeans == 0 , 1], c = 'cyan', label = 'Cluster 5')
plt.title("Clients Cluster")
plt.xlabel("Annual Salary")
plt.ylabel("Spending Score")
plt.legend()

