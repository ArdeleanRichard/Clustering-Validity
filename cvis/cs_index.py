"""
CS Index (Chou-Su-Lai) Cluster Validity Index - Optimized Implementation

Reference:
Chou, C.-H., Su, M.-C., & Lai, E. (2004).
A new cluster validity measure and its application to image compression.
Pattern Analysis and Applications, 7(2), 205-220.

Optimizations:
- Vectorized distance calculations using broadcasting
- Efficient pairwise distance computation with scipy
- Eliminated nested loops
"""

import numpy as np
from scipy.spatial.distance import cdist


def cluster_diameter_vectorized(cluster_points: np.ndarray) -> float:
    """
    Calculate cluster diameter efficiently using vectorized operations.

    Args:
        cluster_points: Points belonging to the cluster (n_points, n_features)

    Returns:
        Cluster diameter (average maximum distance from each point)
    """
    n_points = len(cluster_points)

    if n_points == 0 or n_points == 1:
        return 0.0

    # Compute all pairwise distances at once (n_points x n_points)
    distances = cdist(cluster_points, cluster_points, metric='euclidean')

    # Get maximum distance for each point (ignoring self-distance)
    max_distances = np.max(distances, axis=1)

    # Return average of maximum distances
    return np.mean(max_distances)


def cs_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the CS (Chou-Su-Lai) cluster validity index (optimized version).

    The CS index is defined as:
    CS = (1/K) * Σ(D_i / d_i) = Σ(D_i) / Σ(d_i)

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

    # Compute centroids for all clusters at once
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])

    # Compute all pairwise centroid distances at once
    centroid_distances = cdist(centroids, centroids, metric='euclidean')

    # Initialize accumulators
    total_diameter = 0.0
    total_separation = 0.0

    # Calculate components for each cluster
    for i, k in enumerate(unique_labels):
        # Get points in current cluster
        cluster_points = X[labels == k]

        if len(cluster_points) == 0:
            continue

        # Calculate diameter (compactness) - vectorized
        D_k = cluster_diameter_vectorized(cluster_points)

        # Calculate nearest neighbor distance (separation)
        # Get distances to other centroids (exclude self by masking)
        mask = np.ones(n_clusters, dtype=bool)
        mask[i] = False
        d_k = np.min(centroid_distances[i, mask])

        total_diameter += D_k
        total_separation += d_k

    # Return the CS index
    if total_separation == 0:
        return float('inf')

    return total_diameter / total_separation