import numpy as np

from cvis_ours.ad import ArborisDistanceCalculator, _get_centroid_ids_from_data, _get_centroid_id_from_data


def ad_silhouette_score(data, labels, n_neighbors=5):
    """Silhouette score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(data)

    if n_clusters == 1:
        return 0.0

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, n_neighbors=n_neighbors)

    # Find centroids
    centroid_ids = _get_centroid_ids_from_data(data, labels, unique_labels)

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


def ad_davies_bouldin_score(data, labels, n_neighbors=5):
    """Davies-Bouldin score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, n_neighbors=n_neighbors)

    # Find centroids
    centroid_ids = _get_centroid_ids_from_data(data, labels, unique_labels)

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


def ad_calinski_harabasz_score(data, labels, n_neighbors=5):
    """Calinski-Harabasz score using MST distances."""
    n_samples = len(data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST
    dist_calculator = ArborisDistanceCalculator(data, n_neighbors=n_neighbors)

    # Find centroids
    centroid_ids = _get_centroid_ids_from_data(data, labels, unique_labels)

    # Find overall centroid
    overall_centroid_id = _get_centroid_id_from_data(data)

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




def ad_idea(data, labels, n_neighbors=5):
    """
    Optimized version of mst_idea: ratio of max intra-cluster to min inter-cluster distance.
    Lower is better (compact clusters, well-separated).
    """
    if -1 in labels:
        data_nn = data[labels != -1]
        labels_nn = labels[labels != -1]
        silence_percentage = (len(labels) - np.count_nonzero(labels == -1)) / len(labels)
    else:
        data_nn = np.copy(data)
        labels_nn = np.copy(labels)
        silence_percentage = 1

    unique_labels, unique_counts = np.unique(labels_nn, return_counts=True)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        # if there is a single cluster
        return 0.0

    if np.count_nonzero(unique_counts > 1) <= 1:
        # if there is a no/a single cluster with more than 1 point
        return 0.0

    # For within-cluster distances, build separate MSTs for each cluster
    max_intra_dists = []
    for label in unique_labels:
        cluster_data = data_nn[labels_nn == label]
        if len(cluster_data) > n_neighbors:
            cluster_mst = ArborisDistanceCalculator(cluster_data, n_neighbors=n_neighbors)
            cluster_centroid_id = _get_centroid_id_from_data(cluster_data)
            distances = cluster_mst.get_distances_to_point(cluster_centroid_id)
            max_intra_dists.append(np.mean(distances))
        elif 1 < len(cluster_data) <= n_neighbors:
            diff = cluster_data[:, None, :] - cluster_data[None, :, :]
            distances = np.linalg.norm(diff, axis=-1)
            max_intra_dists.append(np.max(distances))
        elif len(cluster_data) <= 1:
            max_intra_dists.append(np.inf) # clusters of size 1 give 0

    # For inter-cluster distances, use full MST
    mst = ArborisDistanceCalculator(data, n_neighbors=n_neighbors)
    centroid_ids = _get_centroid_ids_from_data(data, labels, unique_labels)

    # Find minimum inter-cluster distance
    # min_inter_dists = []
    # for i in range(n_clusters):
    #     min_inter_dist = np.inf
    #     for j in range(n_clusters):
    #         if i != j:
    #             dist = mst.get_distance(centroid_ids[i], centroid_ids[j])
    #             min_inter_dist = min(min_inter_dist, dist)
    #     min_inter_dists.append(min_inter_dist)

    min_inter_dists = []
    for i, label_i in enumerate(unique_labels):
        min_inter_dist = np.inf
        for j, label_j in enumerate(unique_labels):
            if i != j:
                distances_to_j = mst.get_distances_to_point(centroid_ids[j])
                cluster_i_distances_to_j = distances_to_j[labels == label_i]

                dist = np.mean(cluster_i_distances_to_j)
                min_inter_dist = min(min_inter_dist, dist)
        min_inter_dists.append(min_inter_dist)


    # max intra dist for each cluster
    max_intra_dists = np.array(max_intra_dists)
    # min inter dists (to closest cluster) for each cluster
    min_inter_dists = np.array(min_inter_dists)

    mask = (max_intra_dists != np.inf) & (min_inter_dists != np.inf)
    max_intra_dists = max_intra_dists[mask]
    min_inter_dists = min_inter_dists[mask]

    # return np.sum(min_inter_dists / max_intra_dists) * silence_percentage
    return np.prod(min_inter_dists / max_intra_dists) ** silence_percentage


def main_compare_performance():
    from time import time
    from load_datasets import create_data4
    from sklearn.preprocessing import MinMaxScaler

    for n in [500, 1000, 2000]:
        print(f"\nDataset size: {n} samples")
        X, labels = create_data4(n)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        k = 5
        # Silhouette score
        start = time()
        score = ad_silhouette_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Silhouette: {score:.4f} in {timee:.3f}s")

        # Davies-Bouldin score
        start = time()
        score = ad_davies_bouldin_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Davies-Bouldin: {score:.4f} in {timee:.3f}s")

        # Calinski-Harabasz score
        start = time()
        score = ad_calinski_harabasz_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Calinski-Harabasz: {score:.4f} in {timee:.3f}s")


if __name__ == "__main__":
    pass

    # main_compare_performance()
