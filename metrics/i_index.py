"""
Implementation of I index (Higher better)
github.com/hj-n/pyivm/blob/master/pyivm/ivm/i_index.py

"""

import numpy as np

def i_index(X, labels):
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	## compute the centroids of each cluster
	centroids= np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = centroid(X[labels == i, :])

	## compute the sum of distances of each point to its centroid
	dist_sum_to_centroids = 0
	for i in range(n_clusters):
		dists_squared = np.square(X[labels == i, :] - centroids[i, :])
		dists_row_sum = np.sqrt(np.sum(dists_squared, axis=1))
		dist_sum_to_centroids += np.sum(dists_row_sum)

	## compute the centroid of the whole data set
	centroid_whole = centroid(X)

	## compute the sum of distances to the centroid of the whole data set
	dist_squared_whole = np.square(X - centroid_whole)
	dist_row_sum_whole = np.sqrt(np.sum(dist_squared_whole, axis=1))
	dist_sum_whole = np.sum(dist_row_sum_whole)

	### compute compactness
	compactness =  (dist_sum_whole / dist_sum_to_centroids) / n_clusters

	max_dist = 0
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = euc_dis(centroids[i, :], centroids[j, :])
			if dist > max_dist:
				max_dist = dist
	separability = max_dist

	power = 2

	return (compactness * separability) ** power

def centroid(X):
	return np.mean(X, axis=0)

def euc_dis(vec_1, vec_2):
	return np.linalg.norm(vec_1 - vec_2)






