import numpy as np
import matplotlib.pyplot as plt

from load_datasets import create_data2, create_data1
from ours.ed import centre_from_data, edging_distance
from constants import LABEL_COLOR_MAP


class ED_KMeansClustering:
    def __init__(self, X, num_clusters, neighbors=3, lookahead=3):
        self.K = num_clusters
        self.n_examples = X.shape[0]
        self.n_features = X.shape[1]

        self.max_iterations = 20

        self.neighbors = neighbors
        self.lookahead = lookahead


    def fit(self, X):
        # centroids = self.initialize_random_centroids(X)
        centroids = self.initialize_kmeans_pp(X)

        clusters = self.reassign_points(X, centroids)


        prev_error = float('inf')  # Initialize previous error as infinity
        for it in range(self.max_iterations):
            clusters = self.reassign_points(X, centroids)

            previous_centroids = centroids
            centroids = self.compute_new_centroids(clusters, X)

            error = self.calculate_error(clusters, centroids, X)
            if error >= prev_error:
                break
            else:
                best_centroids = centroids.copy()  # Update the best centroids if error decreases

                # print(f"Iteration {it} - error {error} from prev {prev_error}")

                prev_error = error

                # diff = centroids - previous_centroids
                #
                # if not diff.any():
                #     break

            # label_color = [LABEL_COLOR_MAP[i] for i in clusters]
            # plt.title(f"NewK-Means Iteration {it}")
            # plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', s=25, alpha=0.75)
            # plt.scatter(centroids[:, 0], centroids[:, 1], c='white', marker='X', s=200, edgecolors='k', alpha=1)
            # plt.show()
            # plt.close()

        # print()

        clusters = self.reassign_points_by_paths_in_cluster(X, clusters, centroids)
        return clusters, best_centroids

    def calculate_error(self, clusters, centroids, X):
        error = 0
        for k in range(self.K):
            cluster_points = X[clusters==k]
            if len(cluster_points) > 0:
                for cluster_point in cluster_points:
                    error += edging_distance(X, cluster_point, centroids[k], n_neighbours=self.neighbors, lookahead=self.lookahead)
        return error


    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.n_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.n_examples))]
            centroids[k] = centroid

        return centroids


    def initialize_kmeans_pp(self, X):
        centroids = np.zeros((self.K, self.n_features))

        # Select the first centroid randomly from the dataset
        centroids[0] = X[np.random.choice(range(self.n_examples))]

        # For the remaining centroids
        for k in range(1, self.K):
            # Calculate the squared distances from each point to the nearest centroid
            # distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:n_neighbours]]) for x in X])

            distances = np.array([min([edging_distance(X, x, c, n_neighbours=self.neighbors, lookahead=self.lookahead) for c in centroids[:k]]) for x in X])
            # Choose the next centroid with probability proportional to its squared distance
            probabilities = distances / np.sum(distances)
            centroids[k] = X[np.random.choice(range(self.n_examples), p=probabilities)]

        return centroids


    def reassign_points(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = np.zeros((len(X), ))

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            # closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            dist_to_centres = np.array([edging_distance(X, point, centroid, n_neighbours=self.neighbors, lookahead=self.lookahead) for centroid in centroids])

            closest_centroid = np.argmin(dist_to_centres)
            clusters[point_idx] = closest_centroid

        return clusters


    def reassign_points_by_paths_in_cluster(self, X, old_clusters, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = np.zeros((len(X), ))

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            # closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            dist_to_centres = np.array([edging_distance(np.vstack((X[old_clusters == id], point)), point, centroid, n_neighbours=self.neighbors, lookahead=self.lookahead)
                                        for id, centroid in enumerate(centroids)])

            closest_centroid = np.argmin(dist_to_centres)
            clusters[point_idx] = closest_centroid

        return clusters


    def compute_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.n_features))
        for idc in np.unique(clusters).astype(int):
            new_centroid = centre_from_data(X[clusters==idc])
            centroids[idc] = new_centroid

        return centroids







if __name__ == "__main__":
    np.random.seed(10)
    # num_clusters = 3
    # X, y = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)
    num_clusters = 2
    X, y = create_data1(n_samples=1000)
    # X, y = create_data2(n_samples=1000)

    newKmeans = ED_KMeansClustering(X, num_clusters)
    y_pred = newKmeans.fit(X)

    label_color = [LABEL_COLOR_MAP[i] for i in y_pred]
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o', edgecolors='k', s=25)
    plt.show()