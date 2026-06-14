def calculate_duda_hart_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    # Find the unique cluster labels
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Duda-Hart index is undefined when y_pred has only 1 cluster.")

    # Track overall sums and counts to replicate the exact average-of-averages logic
    intra_cluster_distances = 0.0
    inter_cluster_distances = 0.0

    # Cache masks to avoid re-evaluating y_pred == label multiple times
    masks = [y_pred == label for label in unique_labels]
    counts = [np.count_nonzero(m) for m in masks]

    # Pre-calculate intra-cluster means to avoid repeating work
    intra_means = []
    for i, label in enumerate(unique_labels):
        n_i = counts[i]
        if n_i <= 1:
            # Distance matrix for 1 or 0 elements has a mean of 0.0
            intra_means.append(0.0)
        else:
            # Only compute pairwise distances WITHIN this specific cluster
            X_cluster = X[masks[i]]
            # pdist computes the upper triangle. The full cdist matrix includes
            # the diagonal (zeros) and the symmetric lower triangle.
            intra_sum = np.sum(pdist(X_cluster)) * 2
            intra_mean = intra_sum / (n_i * n_i)
            intra_means.append(intra_mean)

    intra_cluster_distances = sum(intra_means)

    # Compute inter-cluster distances efficiently
    # Distance between cluster i and cluster j
    for i in range(len(unique_labels)):
        n_i = counts[i]
        X_i = X[masks[i]]

        inter_mean_sum_for_cluster_i = 0.0
        for j in range(len(unique_labels)):
            if i == j:
                continue
            n_j = counts[j]
            X_j = X[masks[j]]

            # Compute cross-distances between cluster i and cluster j
            inter_sum = np.sum(cdist(X_i, X_j))
            inter_mean_sum_for_cluster_i += inter_sum / (n_i * n_j)

        inter_cluster_distances += inter_mean_sum_for_cluster_i

    # Calculate the Duda index
    result = intra_cluster_distances / inter_cluster_distances
    return result


import numpy as np
from scipy.spatial.distance import cdist


def calculate_hartigan_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    """
    Compute the Hartigan Index (Optimized for High Dimensions).
    """
    centroids, _ = compute_barycenters(X, y_pred)
    unique_labels = np.unique(y_pred)
    num_clusters = len(unique_labels)

    if num_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Hartigan Index is undefined when y_pred has only 1 cluster.")

    # 1. Precompute a complete distance matrix between ALL centroids
    # This replaces the costly 'np.delete' loop entirely
    centroid_distances = cdist(centroids, centroids, metric='euclidean')

    # Fill diagonal with infinity so a centroid doesn't pick itself as its closest neighbor
    np.fill_diagonal(centroid_distances, np.inf)

    # Get the index of the closest other centroid for all clusters simultaneously
    closest_centroid_indices = np.argmin(centroid_distances, axis=1)

    hi = 0.0

    # 2. Optimized Cluster Loop
    # Mapping directly over the actual unique labels present in y_pred
    for idx, label in enumerate(unique_labels):
        cluster_data = X[y_pred == label]
        cluster_centroid = centroids[idx]

        # Get the pre-calculated closest alternative centroid
        closest_other_idx = closest_centroid_indices[idx]
        closest_other_centroid = centroids[closest_other_idx]

        # Stack the two centroids together into a (2, 5000) matrix
        # This allows us to make ONE highly optimized cdist call instead of TWO separate ones
        both_centroids = np.vstack([cluster_centroid, closest_other_centroid])

        # Compute distances to both target centroids simultaneously at C-speed
        dists_sq = cdist(cluster_data, both_centroids, metric='euclidean') ** 2

        # Slice back the sums directly out of the resulting matrix
        sum_distances_within_cluster = np.sum(dists_sq[:, 0])
        sum_distances_to_closest_other_cluster = np.sum(dists_sq[:, 1])

        hi += sum_distances_within_cluster / sum_distances_to_closest_other_cluster

    return hi