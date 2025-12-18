"""
Optimized implementation of VIASCKDE
https://github.com/senolali/VIASCKDE/blob/main/VIASCKDE.py
https://www.researchgate.net/publication/361177430_VIASCKDE_Index_A_Novel_Internal_Cluster_Validity_Index_for_Arbitrary-Shaped_Clusters_Based_on_the_Kernel_Density_Estimation
https://onlinelibrary.wiley.com/doi/10.1155/2022/4059302

if you use the code, please cite the article given below:

    Ali Şenol, "VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary-Shaped
    Clusters Based on the Kernel Density Estimation", Computational Intelligence and Neuroscience,
    vol. 2022, Article ID 4059302, 20 pages, 2022. https://doi.org/10.1155/2022/4059302
"""

import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings("ignore")


def VIASCKDE(X, labels, kernel='gaussian', b_width=0.05):
    num_k = np.unique(labels)

    if len(num_k) <= 1:
        return float("nan")

    # Compute KDE once for all data
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    # Build KDTree once for all data
    kdtree_all = KDTree(X)

    total_weighted_score = 0.0
    total_count = 0

    for i in num_k:
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]
        other_indices = np.where(~cluster_mask)[0]

        data_of_cluster = X[cluster_mask]
        data_of_not_its = X[~cluster_mask]
        isos = iso[cluster_mask]

        # Normalize isos
        iso_min = isos.min()
        iso_max = isos.max()
        if iso_max > iso_min:
            isos = (isos - iso_min) / (iso_max - iso_min)
        else:
            isos = np.zeros_like(isos)

        # Build KDTree for cluster and other clusters
        kdtree_cluster = KDTree(data_of_cluster)
        kdtree_other = KDTree(data_of_not_its)

        # Query distances for all points at once
        # For within-cluster: k=2 to get nearest neighbor excluding self
        dist_within, _ = kdtree_cluster.query(data_of_cluster, k=2)
        a = dist_within[:, 1]  # Second closest is the nearest neighbor (first is self)

        # For other clusters: k=1 to get nearest neighbor
        b, _ = kdtree_other.query(data_of_cluster, k=1)
        b = b.ravel()

        # Compute ASC for all points in cluster at once
        max_ab = np.maximum(a, b)
        ASC = ((b - a) / max_ab) * isos

        # Accumulate weighted sum
        cluster_count = len(ASC)
        cluster_mean = ASC.mean()

        total_weighted_score += cluster_count * cluster_mean
        total_count += cluster_count

    viasc = total_weighted_score / total_count
    return viasc