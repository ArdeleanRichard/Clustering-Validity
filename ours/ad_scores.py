import numpy as np
from collections import defaultdict, deque
import heapq

from sklearn.neighbors import NearestNeighbors

from ours.ad import ArborisDistanceCalculator, _get_find_cluster_centroids_ids, _get_centroid_id_from_data_fast


def ad_silhouette_score(data, labels, k=5):
    """Silhouette score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(data)

    if n_clusters == 1:
        return 0.0

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, k=k)

    # Find centroids
    centroid_ids = _get_find_cluster_centroids_ids(data, labels, unique_labels)

    # Get distances from all centroids to all points efficiently
    distance_matrix = dist_calculator.get_distances_to_multiple(centroid_ids).T

    # Compute intra-cluster distances (distance to own cluster centroid)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    cluster_indices = np.array([label_to_idx[label] for label in labels])
    intra_distances = distance_matrix[np.arange(n_samples), cluster_indices]

    # Compute inter-cluster distances (minimum distance to other cluster centroids)
    distance_matrix_masked = distance_matrix.copy()
    distance_matrix_masked[np.arange(n_samples), cluster_indices] = np.inf      # Set own cluster distance to inf
    inter_distances = np.min(distance_matrix_masked, axis=1)

    # Compute silhouette coefficients
    max_distances = np.maximum(intra_distances, inter_distances)
    silhouette_coefficients = np.where(
        max_distances > 0,
        (inter_distances - intra_distances) / max_distances,
        0.0
    )

    return np.mean(silhouette_coefficients)


def ad_davies_bouldin_score(data, labels, k=5):
    """Davies-Bouldin score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, k=k)

    # Find centroids
    centroid_ids = _get_find_cluster_centroids_ids(data, labels, unique_labels)

    # Compute inter-cluster distances (between centroids)
    cluster_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = dist_calculator.get_distance(centroid_ids[i], centroid_ids[j])
            cluster_distances[i, j] = dist
            cluster_distances[j, i] = dist

    # Compute intra-cluster scatter (mean distance to centroid)
    cluster_scatter = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) > 0:
            distances = dist_calculator.get_distances_to_point(centroid_ids[i])
            cluster_scatter[i] = np.mean(distances[cluster_indices])

    # Compute Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_similarity = -np.inf
        for j in range(n_clusters):
            if i != j:
                similarity = (cluster_scatter[i] + cluster_scatter[j]) / cluster_distances[i, j]
                max_similarity = max(max_similarity, similarity)
        if max_similarity > -np.inf:
            db_index += max_similarity

    return db_index / n_clusters


def ad_calinski_harabasz_score(data, labels, k=5):
    """Calinski-Harabasz score using MST distances."""
    n_samples = len(data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, k=k)

    # Find centroids
    centroid_ids = _get_find_cluster_centroids_ids(data, labels, unique_labels)

    # Find overall centroid
    overall_centroid_id = _get_centroid_id_from_data_fast(data)

    # Between-cluster sum of squares
    between_ss = 0.0
    for i, label in enumerate(unique_labels):
        cluster_size = np.sum(labels == label)
        dist = dist_calculator.get_distance(overall_centroid_id, centroid_ids[i])
        between_ss += dist * cluster_size

    # Within-cluster sum of squares
    within_ss = 0.0
    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        distances = dist_calculator.get_distances_to_point(centroid_ids[i])
        within_ss += np.sum(distances[cluster_indices])

    # Calinski-Harabasz index
    if within_ss == 0 or n_clusters == 1:
        return 0.0

    ch_index = (between_ss / (n_clusters - 1)) / (within_ss / (n_samples - n_clusters))
    return ch_index



def compute_purity(data, labels, k=5, mode='euclidean'):
    """
    Compute per-point purity: how many of the k nearest neighbors differ in label.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
    labels : 1d array-like, shape (n_samples,)
    k : int
        Number of neighbors to consider (neighbors excluded self).
    mode : {'euclidean'}
        'euclidean' uses Euclidean k-NN (fast via sklearn where available).
    distance_calculator : ArborisDistanceCalculator or None
        If mode == 'mst' you can pass a pre-built DistanceCalculator to avoid rebuilding.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    n_samples = len(data)

    if k <= 0:
        raise ValueError("k must be >= 1")

    # Find k nearest neighbors for each point
    if mode == 'euclidean':
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
        dists, indices = nn.kneighbors(data)
        # indices includes self at position 0
        neighbors = indices[:, 1:k + 1]
    else:
        raise ValueError("mode must be 'euclidean' or 'mst'")

    # Count different-label neighbors
    diff_counts = np.sum(labels[neighbors] != labels[:, np.newaxis], axis=1)
    diff_fractions = diff_counts / float(k)

    # Summary stats
    same_fractions = 1.0 - diff_fractions

    unique_labels = np.unique(labels)
    label_purity = {}
    cluster_sizes = {}
    for ul in unique_labels:
        mask = labels == ul
        size = np.sum(mask)
        cluster_sizes[ul] = int(size)
        if np.sum(mask) > 0:
            # value = np.exp(np.mean(np.log(same_fractions[mask] + 1e-10)))
            # value = np.mean(same_fractions[mask])

            vals = np.sort(same_fractions[mask])  # ascending: worst -> best
            # weights: give largest weight to worst (first) and smallest to best (last)
            # no hyperparameter: integer ranks reversed
            weights = np.arange(size, 0, -1)  # e.g., [n, n-1, ..., 1]
            value = float(np.dot(vals, weights) / float(weights.sum()))
            label_purity[ul] = np.clip(value, 0.0, 1.0)
        else:
            label_purity[ul] = np.nan

    total = 0.0
    min_purity = np.inf
    for ul, pur in label_purity.items():
        total += pur * cluster_sizes[ul]
        min_purity = min(min_purity, pur)
    global_purity = float(total / float(len(labels)))

    return min_purity

def mst_separation_ratio(data, labels, k=5):
    """
    Optimized version of mst_idea: ratio of max intra-cluster to min inter-cluster distance.
    Lower is better (compact clusters, well-separated).
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # For within-cluster distances, build separate MSTs for each cluster
    max_intra_dist = 0.0
    max_intra_dists = []
    for label in unique_labels:
        cluster_data = data[labels == label]
        if len(cluster_data) > 1:
            cluster_mst = ArborisDistanceCalculator(cluster_data, k=k)
            cluster_centroid_id = _get_centroid_id_from_data_fast(cluster_data)

            # Maximum distance from centroid to any point in cluster
            distances = cluster_mst.get_distances_to_point(cluster_centroid_id)
            max_intra_dist = max(max_intra_dist, np.max(distances))
            max_intra_dists.append(np.max(distances))

    # For inter-cluster distances, use full MST
    mst = ArborisDistanceCalculator(data, k=k)

    # Find centroids
    centroid_ids = _get_find_cluster_centroids_ids(data, labels, unique_labels)

    # Find minimum inter-cluster distance
    min_inter_dists = []
    for i in range(n_clusters):
        min_inter_dist = np.inf
        for j in range(n_clusters):
            if i != j:
                dist = mst.get_distance(centroid_ids[i], centroid_ids[j])
                min_inter_dist = min(min_inter_dist, dist)
        min_inter_dists.append(min_inter_dist)

    purity = compute_purity(data, labels, k)
    # print(min_inter_dist, max_intra_dist, purity)

    # return min_inter_dist / max_intra_dist * purity

    max_intra_dists = np.array(max_intra_dists)
    min_inter_dists = np.array(min_inter_dists)

    # print(min_inter_dists, max_intra_dists, purity)

    return np.sum(min_inter_dists / max_intra_dists) # * purity


def compare_performance():
    from time import time
    from load_datasets import create_data4
    from sklearn.preprocessing import MinMaxScaler

    print("Performance Comparison")
    print("=" * 60)

    for n in [500, 1000, 2000]:
        print(f"\nDataset size: {n} samples")
        X, labels = create_data4(n)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        k = 5
        # Silhouette score
        start = time()
        score_opt = ad_silhouette_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Silhouette: {score_opt:.4f} in {time_opt:.3f}s")

        # Davies-Bouldin score
        start = time()
        score_opt = ad_davies_bouldin_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Davies-Bouldin: {score_opt:.4f} in {time_opt:.3f}s")

        # Calinski-Harabasz score
        start = time()
        score_opt = ad_calinski_harabasz_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Calinski-Harabasz: {score_opt:.4f} in {time_opt:.3f}s")


if __name__ == "__main__":
    pass

    # compare_performance()
