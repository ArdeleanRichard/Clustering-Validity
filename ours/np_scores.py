import numpy as np
from sklearn import datasets
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from load_datasets import create_data3, create_data6, create_data7, create_data4, create_data2, create_data1, create_data5


def np_calinski_harabasz_score(X, labels):
    """
    Calculate the Calinski-Harabasz index for a given clustering.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - ch_index: float
        The Calinski-Harabasz index.
    """
    n_samples, n_features = X.shape
    k = np.max(labels) + 1  # Number of clusters

    # Calculate cluster means
    cluster_means = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])

    # Calculate overall mean
    overall_mean = np.mean(X, axis=0)

    # Calculate between-cluster sum of squares
    between_cluster_ss = np.sum([np.sum((cluster_means[i] - overall_mean) ** 2) * np.sum(labels == i) for i in range(k)])

    # Calculate within-cluster sum of squares
    within_cluster_ss = np.sum([np.sum((X[labels == i] - cluster_means[i]) ** 2) for i in range(k)])

    # Calculate Calinski-Harabasz index
    ch_index = (between_cluster_ss / (k - 1)) / (within_cluster_ss / (n_samples - k))

    return ch_index


def np_davies_bouldin_score(X, labels):
    """
    Calculate the Davies-Bouldin index for a given clustering.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - db_index: float
        The Davies-Bouldin index.
    """
    n_samples, n_features = X.shape
    k = np.max(labels) + 1  # Number of clusters

    # Calculate cluster means
    cluster_means = np.array([np.mean(X[labels == i], axis=0) for i in range(k)])

    # Calculate cluster distances
    cluster_distances = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                cluster_distances[i, j] = np.linalg.norm(cluster_means[i] - cluster_means[j])

    # Calculate cluster-wise scatter
    cluster_scatter = np.array([np.mean([np.linalg.norm(X[m] - cluster_means[i]) for m in range(n_samples) if labels[m] == i]) for i in range(k)])

    # Calculate Davies-Bouldin index
    db_index = 0.0
    for i in range(k):
        max_similarity = -np.inf
        for j in range(k):
            if i != j:
                similarity = (cluster_scatter[i] + cluster_scatter[j]) / cluster_distances[i, j]
                if similarity > max_similarity:
                    max_similarity = similarity
        db_index += max_similarity
    db_index /= k

    return db_index


def np_silhouette_score(X, labels):
    """
    Calculate the silhouette score for a given clustering.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - silhouette_avg: float
        The silhouette score.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0.0

    # Compute the mean silhouette score for all samples
    silhouette_values = np.zeros(n_samples)
    for i in range(n_samples):
        # Calculate the mean intra-cluster distance (a_i)
        cluster_label = labels[i]
        cluster_points = X[labels == cluster_label]
        a_i = np.mean(np.linalg.norm(cluster_points - X[i], axis=1))

        # Calculate the mean nearest-cluster distance (b_i)
        b_i = np.inf
        for j in range(n_clusters):
            if j != cluster_label:
                other_cluster_points = X[labels == j]
                dist = np.mean(np.linalg.norm(other_cluster_points - X[i], axis=1))
                b_i = min(b_i, dist)

        # Compute silhouette value for sample i
        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    silhouette_avg = np.mean(silhouette_values)
    return silhouette_avg



def np_silhouette_score2(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters == 1:
        return 0

    cluster_means = np.array([np.mean(X[labels == i], axis=0) for i in unique_labels])

    # Compute the distance matrix between samples and cluster means
    distance_matrix = np.linalg.norm(X[:, np.newaxis] - cluster_means, axis=2)

    # Compute the intra-cluster distances
    intra_cluster_distances = np.zeros(n_samples)
    for i in range(num_clusters):
        intra_cluster_distances[labels == unique_labels[i]] = np.mean(distance_matrix[labels == unique_labels[i], i])

    # Compute the inter-cluster distances for each sample
    inter_cluster_distances = np.min(distance_matrix + np.where(labels[:, np.newaxis] == unique_labels, np.inf, 0), axis=1)

    # Compute the silhouette coefficient for each sample
    silhouette_coefficients = (inter_cluster_distances - intra_cluster_distances) / np.maximum(inter_cluster_distances, intra_cluster_distances)

    # Compute the mean silhouette score over all samples
    mean_silhouette_score = np.mean(silhouette_coefficients)

    return mean_silhouette_score


if __name__ == '__main__':
    # X, y = create_data_circles()
    # X, y = create_data_moons()
    # X, y = create_data_blobs()
    # X, y = create_data_elongated_close_blobs()
    # X, y = create_data_diff_density_blobs()
    # X, y = create_data_nostructure()
    X, y = create_data6()
    # X, y = create_data2()

    print(calinski_harabasz_score(X, y))
    print(np_calinski_harabasz_score(X, y))
    print()
    print(davies_bouldin_score(X, y))
    print(np_davies_bouldin_score(X, y))
    print()
    print(silhouette_score(X, y))
    print(np_silhouette_score(X, y))
    print(np_silhouette_score2(X, y))
    print()
