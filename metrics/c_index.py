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
from scipy.spatial.distance import pdist, squareform


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

    # Convert labels to numpy array once
    labels = np.asarray(labels)

    # Calculate all pairwise distances once
    distances = pdist(X, metric='euclidean')

    # Calculate within-cluster statistics efficiently
    Sw, Nw = calc_sw_and_nw(distances, labels)

    # Sum of Nw smallest and largest distances
    Smin, Smax = calc_smin_smax(distances, Nw)

    # Calculate CIndex
    cindex = (Sw - Smin) / (Smax - Smin)
    return cindex


def calc_sw_and_nw(distances, labels):
    """Calculate both Sw (sum of within-cluster distances) and Nw (number of
    within-cluster pairs) in a single pass.

    inputs
    -------
    distances : (np.ndarray) condensed distance matrix from pdist
    labels : (np.ndarray) cluster labels

    outputs
    -------
    Sw : (float) sum of within-cluster distances
    Nw : (int) total number of within-cluster pairs
    """
    n = len(labels)

    # Convert condensed distance matrix to square form for easier indexing
    dist_matrix = squareform(distances)

    # Create a mask for within-cluster pairs
    # This is more efficient than looping through clusters
    label_match = labels[:, None] == labels[None, :]

    # Get upper triangle only (to avoid double counting)
    triu_indices = np.triu_indices(n, k=1)
    within_cluster_mask = label_match[triu_indices]

    # Extract within-cluster distances
    within_distances = dist_matrix[triu_indices][within_cluster_mask]

    Sw = np.sum(within_distances)
    Nw = len(within_distances)

    return Sw, Nw


def calc_smin_smax(distances, n_incluster_pairs):
    """Calculate Smin and Smax,
    Smax is the Sum of Nw largest distances between all pairs of points
    in the entire data set, and
    Smin is the Sum of Nw smallest distances between all pairs of points
    in the entire data set

    inputs
    -------
    distances : (np.ndarray) of shape [m,] where m is the pairwise distances
        between each point [i,k] being clustered. m is calculated as
        m = n_points * (n_points - 1) / 2 where n_points is the number of points
        in the entire dataset
    n_incluster_pairs : (int) Total number of pairs of observations belonging
        to the same cluster
    outputs
    -------
    Smin, Smax : (float)
    """
    # Use partition instead of full sort for better performance
    # partition finds the k smallest/largest elements without fully sorting
    if n_incluster_pairs >= len(distances):
        # Edge case: all pairs are within clusters
        Smin = np.sum(distances)
        Smax = np.sum(distances)
    else:
        # Get Nw smallest values efficiently
        smallest_indices = np.argpartition(distances, n_incluster_pairs - 1)[:n_incluster_pairs]
        Smin = np.sum(distances[smallest_indices])

        # Get Nw largest values efficiently
        largest_indices = np.argpartition(distances, -n_incluster_pairs)[-n_incluster_pairs:]
        Smax = np.sum(distances[largest_indices])

    return Smin, Smax