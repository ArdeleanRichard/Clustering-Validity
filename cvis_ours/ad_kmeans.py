import numpy as np
import matplotlib.pyplot as plt
from cvis_ours.ad import ArborisDistanceCalculator, _get_centroid_id_from_data_fast
from constants import LABEL_COLOR_MAP
from load_datasets import create_data1, create_data7, create_data6


class AD_KMeansClustering:
    def __init__(self, X, n_clusters, n_neighbors=5, verbose=False):
        """
        KMeans clustering using MST-based distances.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            The data to cluster
        num_clusters : int
            Number of clusters (K)
        k_neighbors : int
            Number of nearest neighbors for MST construction
        """
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]

        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

        self.max_iterations = 20
        self.best_centroids = None
        self.best_centroid_ids = None
        self.labels = None
        self.n_convergence = 0

        self.verbose = verbose

    def fit(self, X):
        """
        Fit the MST-KMeans model to the data.

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            The data to cluster

        Returns:
        --------
        labels : ndarray, shape (n_samples,)
            Cluster labels for each point
        """
        # Build MST for the entire dataset
        self.dist_calculator = ArborisDistanceCalculator(X, n_neighbors=self.n_neighbors)

        # Initialize centroids
        centroid_ids = self.initialize_kmeans_pp(X)

        prev_error = float('inf')
        self.best_centroid_ids = centroid_ids.copy()

        for it in range(self.max_iterations):
            labels = self.reassign_points(centroid_ids)

            # Compute new centroids
            centroids, centroid_ids = self.compute_new_centroids(labels, X)

            # Calculate error
            error = self.calculate_error(labels, centroid_ids)

            if error >= prev_error:
                self.n_convergence = it
                if self.verbose:
                    print(f"Converged at iteration {it} due to {it-1} (error {prev_error}) < {it} (error {error})")
                break
            else:
                self.best_centroid_ids = centroid_ids.copy()
                prev_error = error

                if self.verbose:
                    print(f"Iteration {it} - error: {error:.4f} (prev: {prev_error:.4f})")

        # Use the best centroid IDs (indices) for final assignment
        self.labels = self.reassign_points(self.best_centroid_ids)

        return self.labels

    def predict(self):
        return self.labels

    def fit_predict(self, X):
        self.fit(X)
        return self.predict()

    def calculate_error(self, labels, centroid_ids):
        """Calculate total MST distance error."""
        error = 0
        for k in range(self.n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Get distances from centroid to all cluster points
                distances = self.dist_calculator.get_distances_to_point(centroid_ids[k])
                error += np.sum(distances[cluster_indices])

        return error

    def initialize_kmeans_pp(self, X):
        """Initialize centroids using k-means++ strategy with MST distances."""
        centroids = np.zeros((self.n_clusters, self.n_features))
        centroid_ids = np.zeros(self.n_clusters, dtype=int)

        # Select first centroid randomly
        first_idx = np.random.choice(range(self.n_examples))
        centroids[0] = X[first_idx]
        centroid_ids[0] = first_idx

        # For remaining centroids
        for k in range(1, self.n_clusters):
            # Get distances from all points to nearest existing centroid
            min_distances = np.full(self.n_examples, np.inf)

            for point_idx in range(self.n_examples):
                for c_id in centroid_ids[:k]:
                    dist = self.dist_calculator.get_distance(point_idx, c_id)
                    min_distances[point_idx] = min(min_distances[point_idx], dist)

            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities /= np.sum(probabilities)

            next_idx = np.random.choice(range(self.n_examples), p=probabilities)
            centroids[k] = X[next_idx]
            centroid_ids[k] = next_idx

        return centroid_ids

    def reassign_points(self, centroid_ids):
        """Assign each point to nearest centroid using MST distances."""
        # Get distances from all centroids to all points efficiently
        distance_matrix = self.dist_calculator.get_distances_to_multiple(centroid_ids).T

        # Assign to nearest centroid
        labels = np.argmin(distance_matrix, axis=1)

        return labels

    def compute_new_centroids(self, labels, X):
        """Compute new centroids as the point minimizing sum of distances."""
        centroids = np.zeros((self.n_clusters, self.n_features))
        centroid_ids = np.zeros(self.n_clusters, dtype=int)

        for k in range(self.n_clusters):
            cluster_mask = labels == k
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Find centroid using the fast method
                centroid_id = _get_centroid_id_from_data_fast(
                    X[cluster_mask],
                    indices=cluster_indices
                )
                centroids[k] = X[centroid_id]
                centroid_ids[k] = centroid_id
            else:
                # Keep previous centroid if cluster is empty
                centroids[k] = X[centroid_ids[k]]

        return centroids, centroid_ids


if __name__ == "__main__":
    # X, y = create_data1(n_samples=1000)
    # X, y = create_data2(n_samples=1000)
    X, y = create_data6(n_samples=1000)
    # X, y = create_data7(n_samples=1000)

    newKmeans = AD_KMeansClustering(X, len(np.unique(y)))
    y_pred = newKmeans.fit(X)
    print(y_pred)

    label_color = [LABEL_COLOR_MAP[i] for i in y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o', edgecolors='k', s=25)
    plt.show()