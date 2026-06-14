"""
Optimized Implementation of CIndex
https://github.com/johnvorsten/py_cindex/tree/master

Cindex =
Sw − Smin
Smax − Smin , Smin ≠ Smax, Cindex ∈ (0, 1), (6)

Smin = is the sum of the Nw smallest distances between all the pairs of points
in the entire data set (there are Nt such pairs);

Smax = is the sum of the Nw largest distances between all the pairs of points
in the entire data set.

Citation:
A General Statistical Framework for Assessing Categorical Clustering in Free Recall."
Psychological Bulletin, 83(6), 1072–1080
"""
import numpy as np
from scipy.spatial.distance import pdist


def c_index(X, labels):
    """Calculate CIndex
    inputs
    -------
    X : (np.ndarray) an (n x m) array where n is the number of examples to cluster
        and m is the feature space of examples
    labels : (np.array) of cluster labels, each labels[i] related to X[i]
        ideally integer type
    output
    -------
    cindex : (float)"""

    labels = np.asarray(labels)

    # Calculate all pairwise distances once (condensed form)
    distances = pdist(X, metric='euclidean')

    # Calculate within-cluster statistics efficiently
    Sw, Nw = calc_sw_and_nw(distances, labels)

    # Sum of Nw smallest and largest distances
    Smin, Smax = calc_smin_smax(distances, Nw)

    cindex = (Sw - Smin) / (Smax - Smin)
    return cindex


def calc_sw_and_nw(distances, labels):
    """Calculate both Sw (sum of within-cluster distances) and Nw (number of
    within-cluster pairs) in a single pass, without expanding the condensed
    distance matrix to square form.

    For each cluster, the global indices of its members are used to directly
    address the condensed distance vector via the formula:
        condensed_index(i, j) = i * (2n - i - 1) // 2 + j - i - 1,  i < j

    This avoids allocating an n×n square matrix and an n×n label-match matrix,
    which are the dominant memory and time costs for large n.

    inputs
    -------
    distances : (np.ndarray) condensed distance matrix from pdist, shape (n*(n-1)/2,)
    labels : (np.ndarray) cluster labels, shape (n,)

    outputs
    -------
    Sw : (float) sum of within-cluster distances
    Nw : (int) total number of within-cluster pairs
    """
    n = len(labels)
    Sw = 0.0
    Nw = 0

    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue

        # All pairs (i, j) with i < j within this cluster
        ii, jj = np.triu_indices(len(idx), k=1)
        gi, gj = idx[ii], idx[jj]  # global indices into the condensed vector

        # Map global pair (gi, gj) to condensed distance vector index
        condensed_idx = gi * (2 * n - gi - 1) // 2 + gj - gi - 1

        Sw += distances[condensed_idx].sum()
        Nw += len(condensed_idx)

    return Sw, Nw


def calc_smin_smax(distances, n_incluster_pairs):
    """Calculate Smin and Smax.
    Smax is the sum of the Nw largest distances between all pairs of points
    in the entire data set, and
    Smin is the sum of the Nw smallest distances between all pairs of points
    in the entire data set.

    Uses np.argpartition for O(n) selection instead of O(n log n) full sort.

    inputs
    -------
    distances : (np.ndarray) condensed distance vector from pdist, shape (m,)
    n_incluster_pairs : (int) total number of pairs belonging to the same cluster
    outputs
    -------
    Smin, Smax : (float)
    """
    if n_incluster_pairs >= len(distances):
        # Edge case: all pairs are within clusters
        s = np.sum(distances)
        return s, s

    # O(m) partial selection — no full sort needed
    smallest_indices = np.argpartition(distances, n_incluster_pairs - 1)[:n_incluster_pairs]
    Smin = np.sum(distances[smallest_indices])

    largest_indices = np.argpartition(distances, -n_incluster_pairs)[-n_incluster_pairs:]
    Smax = np.sum(distances[largest_indices])

    return Smin, Smax




if __name__ == "__main__":
    import time
    from load_datasets import create_data1, create_data2, create_data3

    X, y = create_data1(10000)
    start = time.time()
    score = c_index(X, y)
    print(f"C: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data2(10000)
    start = time.time()
    score = c_index(X, y)
    print(f"C: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data3(10000)
    start = time.time()
    score = c_index(X, y)
    print(f"C: {score:.3f} in {time.time()-start:.3f}s")
