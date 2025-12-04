"""
Implementation of VIASCKDE
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


def closest_node(n, v):
    kdtree = KDTree(v)
    d, i = kdtree.query(n)
    return d


def VIASCKDE(X, labels, kernel='gaussian', b_width=0.05):
    cluster_ids = np.unique(labels)
    # Kernel Density Estimation
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(X)
    # log-likelihood of each sample under the model
    iso = kde.score_samples(X)

    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(cluster_ids) > 1:
        for i in cluster_ids:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            cluster_isos = iso[labels == i]
            # min max normalization of isos
            cluster_isos = (cluster_isos - min(cluster_isos)) / (max(cluster_isos) - min(cluster_isos))
            for j in range(len(data_of_cluster)):  # for each point j of cluster i
                row = np.delete(data_of_cluster, j, 0)  # exclude the point j
                current_point = data_of_cluster[j]
                # Silhouette type of a/b
                a = closest_node(current_point, row) # a is closest node in cluster
                b = closest_node(current_point, data_of_not_its) # b is closest node of any other cluster
                # Silhouette in KDtree multipled by ISO added to ASC vector for each point
                ASC = np.hstack(
                    (
                        ASC,
                        ((b - a) / max(a, b)) * cluster_isos[j]
                    )
                )
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean())) # ASC mean of cluster

        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]

        viasc = viasc / sum(numC)
    else:
        viasc = float("nan")

    return viasc

def VIASCKDE_DEBUG():
    from sklearn.datasets import make_blobs
    # Create test data: 3 well-separated clusters with 50 points
    np.random.seed(42)
    n_samples_per_cluster = [18, 16, 16]
    centers = [[2, 2], [8, 8], [14, 2]]
    cluster_std = [0.8, 0.9, 0.7]

    X, y = make_blobs(n_samples=n_samples_per_cluster,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=42)

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of points: {len(X)}")
    print(f"Cluster distribution:")
    for i in range(len(centers)):
        print(f"  Cluster {i}: {np.sum(y == i)} points")

    VIASCKDE_score = VIASCKDE(X, y)

    return VIASCKDE_score

if __name__ == "__main__":
    VIASCKDE_DEBUG()