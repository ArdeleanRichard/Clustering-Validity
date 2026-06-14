import scipy.spatial.distance as dist
import numpy as np


import numpy as np
import scipy.spatial.distance as dist


def cop(data, labels, **kwargs):
    """
    Highly optimized COP Index calculation.
    Maintains 100% mathematical parity with the original version.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)

    N = len(data)
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # 1. Map labels to sequential indices for fast grouping
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_to_idx[l] for l in labels])

    # 2. Compute all centroids and cluster sizes simultaneously
    centroids = np.zeros((num_clusters, data.shape[1]))
    cluster_sizes = np.zeros(num_clusters)
    clusters = []

    for i, label in enumerate(unique_labels):
        cluster_data = data[mapped_labels == i]
        clusters.append(cluster_data)
        centroids[i] = np.mean(cluster_data, axis=0)
        cluster_sizes[i] = len(cluster_data)

    # 3. Compute pairwise maximum distances between all clusters upfront
    # max_inter_dist[i, j] will hold the max distance between cluster i and cluster j
    max_inter_dist = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            # cdist computes the full cross-distance matrix between cluster i and j
            dists = dist.cdist(clusters[i], clusters[j], **kwargs)
            max_val = np.max(dists)
            max_inter_dist[i, j] = max_val
            max_inter_dist[j, i] = max_val

    cop_k = 0.0

    # 4. Calculate the COP score for each cluster
    for i in range(num_clusters):
        cluster_k = clusters[i]
        n_k = cluster_sizes[i]

        # Vectorized Intra-Cluster distance from centroid
        # Reshaping centroids[i] ensures scipy treats it as a single row vector
        intra_cdist = np.sum(
            dist.cdist(cluster_k, centroids[i : i + 1], **kwargs)
        )
        intra_cop = intra_cdist / n_k

        # Retrieve the precomputed inter-cluster maximum distances
        # Original code looks at alternative clusters: list(set(labels) ^ {k})
        # Which translates to all clusters except the current one (i)
        mask = np.ones(num_clusters, dtype=bool)
        mask[i] = False
        inter_cop = np.min(max_inter_dist[i, mask])

        cop_k += n_k * (intra_cop / inter_cop)

    return cop_k / N

if __name__ == "__main__":
    import time
    from load_datasets import create_data5, create_data2, create_data3

    X, y = create_data5(1000, 4000)
    start = time.time()
    score = cop(X, y)
    print(f"COP: {score:.3f} in {time.time()-start:.3f}s")
