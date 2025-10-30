import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from load_datasets import create_data1
from constants import LABEL_COLOR_MAP


class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]


    def fit(self, X):
        # centroids = self.initialize_random_centroids(X)
        centroids = self.initialize_kmeans_pp(X)

        best_clusters = None


        prev_error = float('inf')  # Initialize previous error as infinity
        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)



            error = self.calculate_error(clusters, centroids, X)
            if error > prev_error:
                break
            else:
                best_clusters = clusters.copy()  # Update the best centroids if error decreases

                prev_error = error

                diff = centroids - previous_centroids
                if not diff.any():
                    break


        return best_clusters


    def calculate_error(self, clusters, centroids, X):
        error = 0
        for k in range(self.K):
            cluster_points = X[clusters==k]
            if len(cluster_points) > 0:
                error += np.linalg.norm(cluster_points - centroids[k])
        return error


    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids


    def initialize_kmeans_pp(self, X):
        centroids = np.zeros((self.K, self.num_features))

        # Select the first centroid randomly from the dataset
        centroids[0] = X[np.random.choice(range(self.num_examples))]

        # For the remaining centroids
        for k in range(1, self.K):
            # Calculate the squared distances from each point to the nearest centroid
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:k]]) for x in X])
            # Choose the next centroid with probability proportional to its squared distance
            probabilities = distances / np.sum(distances)
            centroids[k] = X[np.random.choice(range(self.num_examples), p=probabilities)]

        return centroids


    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = np.zeros((len(X),))

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            clusters[point_idx] = closest_centroid

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idc in np.unique(clusters).astype(int):
            new_centroid = np.mean(X[clusters==idc], axis=0)
            centroids[idc] = new_centroid

        return centroids


if __name__ == "__main__":
    np.random.seed(10)
    # num_clusters = 3
    # X, y = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)
    num_clusters = 2
    X, y = create_data1(n_samples=1000)

    Kmeans = KMeansClustering(X, num_clusters)
    y_pred = Kmeans.fit(X)

    label_color = [LABEL_COLOR_MAP[i] for i in y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o', edgecolors='k', s=25)
    plt.show()
