from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features only, no labels

wcss = []  # Within-cluster sum of squares

# Try k values from 1 to 25
for i in range(1, 26):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) #n_init is the number of times the k-means algorithm will be run with different centroid seeds
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 26), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)

# 3D scatter plot using the first three features
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for data points, colored by cluster label
scatter = ax.scatter(
    X[:, 0], X[:, 1], X[:, 2],
    c=labels, cmap='viridis', s=50, alpha=0.6
)

# Plot cluster centers
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 2],
    s=200, c='red', marker='*', label='Cluster centers'
)

# Axis labels
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')

# Title
plt.title('3D Cluster Visualization (First 3 Features)')

# Add legend for cluster colors
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)

# Add legend for cluster centers
ax.legend(loc='upper right')

plt.show()