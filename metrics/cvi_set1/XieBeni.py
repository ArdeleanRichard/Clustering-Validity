import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class XieBeniIndex:
    def __init__(self, data, labels):
        """
        Initializes the class with the provided dataframe.

        :param df: pandas DataFrame where the last column is the label (cluster).
        """
        self.data = data
        self.labels = labels

    def _calculate_centroids(self):
        """
        Calculate the centroids of each cluster.

        :return: A dictionary of centroids, keyed by cluster label.
        """
        centroids = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            cluster_data = self.data[self.labels == label]
            centroid = np.mean(cluster_data, axis=0)
            centroids[label] = centroid
        return centroids

    def _calculate_distances(self, centroids):
        """
        Calculate the pairwise distances between points and their respective cluster centroids.

        :param centroids: A dictionary of centroids with cluster labels.
        :return: A numpy array of distances between points and their corresponding cluster centroids.
        """
        distances = np.zeros(len(self.data))
        for i, point in enumerate(self.data):
            cluster_label = self.labels[i]
            centroid = centroids[cluster_label]
            distances[i] = np.linalg.norm(point - centroid)
        return distances

    def _calculate_inter_cluster_distances(self, centroids):
        """
        Calculate the minimum pairwise distance between centroids of different clusters.

        :param centroids: A dictionary of centroids with cluster labels.
        :return: The minimum distance between any two cluster centroids.
        """
        centroid_list = np.array(list(centroids.values()))
        dist_matrix = cdist(centroid_list, centroid_list)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore the diagonal (same centroids)
        return np.min(dist_matrix)

    def run(self):
        """
        Calculate and return the Xie-Beni Index.

        :return: Xie-Beni Index value.
        """
        centroids = self._calculate_centroids()
        distances = self._calculate_distances(centroids)
        inter_cluster_distance = self._calculate_inter_cluster_distances(centroids)

        # Compute the Xie-Beni Index
        compactness = np.sum(distances**2)  # Sum of squared distances within clusters
        index = compactness / inter_cluster_distance**2

        return index
