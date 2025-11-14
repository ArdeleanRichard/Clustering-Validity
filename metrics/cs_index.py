"""
CS Index (Chou-Su-Lai) Cluster Validity Index Implementation

Reference:
Chou, C.-H., Su, M.-C., & Lai, E. (2004).
A new cluster validity measure and its application to image compression.
Pattern Analysis and Applications, 7(2), 205-220.

The CS index evaluates clustering quality by measuring the ratio of
compactness (cluster diameter) to separation (nearest neighbor distance).
Lower values indicate better clustering.
"""

import numpy as np
from typing import Union, Tuple


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point
        point2: Second point

    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def cluster_diameter(cluster_points: np.ndarray) -> float:
    """
    Calculate the diameter of a cluster as the average distance from
    each point to its farthest neighbor within the cluster.

    Args:
        cluster_points: Points belonging to the cluster (n_points, n_features)

    Returns:
        Cluster diameter
    """
    if len(cluster_points) == 0:
        return 0.0

    if len(cluster_points) == 1:
        return 0.0

    # Calculate maximum distance from each point to all other points in cluster
    max_distances = []
    for i, point in enumerate(cluster_points):
        distances = [euclidean_distance(point, other_point) for j, other_point in enumerate(cluster_points) if i != j]
        if distances:
            max_distances.append(max(distances))

    # Diameter is the average of maximum distances
    return np.mean(max_distances) if max_distances else 0.0


def nearest_cluster_distance(centroid: np.ndarray, other_centroids: np.ndarray) -> float:
    """
    Calculate the minimum distance from a cluster centroid to other centroids.

    Args:
        centroid: Current cluster centroid
        other_centroids: Array of other cluster centroids

    Returns:
        Distance to nearest cluster
    """
    if len(other_centroids) == 0:
        return float('inf')

    distances = [euclidean_distance(centroid, other_centroid) for other_centroid in other_centroids]
    return min(distances) if distances else float('inf')


def cs_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the CS (Chou-Su-Lai) cluster validity index.

    The CS index is defined as:
    CS = (1/K) * Σ(D_i / d_i)

    where:
    - K is the number of clusters
    - D_i is the diameter of cluster i (compactness measure)
    - d_i is the distance from cluster i's centroid to its nearest neighbor cluster
      (separation measure)

    Lower CS values indicate better clustering (compact and well-separated clusters).

    Args:
        X: Data points (n_samples, n_features)
        labels: Cluster labels for each point (n_samples,)

    Returns:
        CS index value (lower is better)

    Raises:
        ValueError: If X and labels have incompatible shapes or if there's only one cluster
    """
    # Input validation
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples")

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError("CS index requires at least 2 clusters")

    # Compute centroids
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])

    # Calculate CS index components for each cluster
    D_ks = []
    d_ks = []
    for i, k in enumerate(unique_labels):
        # Get points in current cluster
        cluster_points = X[labels == k]

        if len(cluster_points) == 0:
            continue

        # Calculate diameter (compactness) of cluster k
        D_k = cluster_diameter(cluster_points)

        # Get other centroids for separation calculation
        other_centroids = np.delete(centroids, i, axis=0)

        # Calculate nearest neighbor distance (separation)
        d_k = nearest_cluster_distance(centroids[i], other_centroids)

        D_ks.append(D_k)
        d_ks.append(d_k)

    # CS index is the average of all components
    if len(D_ks) == 0 or len(d_ks) == 0:
        return float('inf')
    else:
        return sum(D_ks) / sum(d_ks)

