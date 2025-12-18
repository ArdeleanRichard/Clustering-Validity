import numpy as np


def imbalance_ratio(X, labels):
    """
    Imbalance Ratio (IR) = majority_class_size / minority_class_size.
    If a class has 0 members, returns np.inf.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    max_count = np.max(counts)
    min_count = np.min(counts)
    if min_count == 0:
        return np.inf

    return max_count / min_count


def overlap_ratio(X, labels, slack=1.2):
    """
    Vectorized computation of the R-value (overlap rate).
    For each point, compute distance to its cluster center and the nearest other center.
    Point is 'overlapping' if dist_to_nearest_other <= slack * dist_to_own.
    Returns fraction of overlapping points.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array (N, dims)")
    unique_labels, inv = np.unique(labels, return_inverse=True)

    # compute centers in the order of unique_labels
    centers = np.vstack([X[labels == lab].mean(axis=0) for lab in unique_labels])

    # distances: shape (N_points, n_centers)
    # broadcasting: points[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    # own center distance for each point
    own_center_idx = inv  # maps each point to index in centers
    idx = np.arange(X.shape[0])
    dist_to_own = dists[idx, own_center_idx]

    # distance to nearest other center: set own center distance to +inf and take min
    dists_other = dists.copy()
    dists_other[idx, own_center_idx] = np.inf
    dist_to_nearest_other = dists_other.min(axis=1)

    # overlapping condition
    overlapping = dist_to_nearest_other <= (slack * dist_to_own)
    overlap_rate = overlapping.sum() / X.shape[0]
    return overlap_rate
