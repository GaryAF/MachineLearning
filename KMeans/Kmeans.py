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
plt.scatter(X[:,0], X[:,1], c=kmeans.predict(X))
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='r')
print(kmeans.score(X))
plt.show()
