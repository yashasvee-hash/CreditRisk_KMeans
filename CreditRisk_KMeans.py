# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:33:33 2021

@author: Yashasvee Shukla
"""

import pandas as pd
df = pd.read_csv('E:\Practice\K-Means\creditRisk\german_credit_data.csv')
df = df.dropna()
df = df.drop(['Unnamed: 0'], axis=1)


df = pd.get_dummies(df, prefix=['Sex', 'Housing', 'Saving account', 'Checking account', 'Purpose'], drop_first = True)


from sklearn.decomposition import PCA
pca = PCA(2)
projected = pca.fit_transform(df)
print(df.shape)
print(projected.shape)



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(projected)
    wcss.append(km.inertia_)
    

plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of centroids')
plt.ylabel('WCSS')
plt.title('Elbow Method for optimal k')
plt.show()



#converting our projected array to pandas df
pca = pd.DataFrame(projected)
pca.columns = ['First component', 'Second component']

#build our algorithm with k=3, train it on pca and make predictions
kmeans = KMeans(n_clusters = 3, random_state=0).fit(pca)
y_kmeans = kmeans.predict(pca)


#plotting the results
plt.scatter(pca['First component'], pca['Second component'], c = y_kmeans, s = 50, alpha = 0.5, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)



