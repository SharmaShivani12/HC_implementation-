'''
In this section we are going to implement the hierarichical clustering algorithm.
Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. 
In general, the strategy is to create a tree of clusters called a dendrogram, where individual elements each start in their own cluster, and pairs of clusters are merged as one moves up the hierarchy.
Two types of HC:Agglomerative (Bottom-Up) and Divisive (Top-Down)
The key operation in hierarchical agglomerative clustering is to repeatedly combine the two nearest clusters into a larger cluster. 
There are different ways to define the distance (or similarity) between clusters, known as linkage criteria.

For dendogram implementation we will use the library :import scipy.cluster.hierarchy
Also we will use the linkage criteria Ward's Method.We are going to perform agglomerative HC(library: from sklearn.cluster import AgglomerativeClustering)
Ward’s Method: This approach minimizes the total within-cluster variance.
 At each step, the pair of clusters with the minimum between-cluster distance are merged.

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#dendogram to find the optimal clusters in the dataset.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers:')
plt.ylabel('Euclidean distances:')
plt.show()

# Using the library of agglomerative method of HC we will train the dataset.
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc== 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

