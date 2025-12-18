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
    num_k = np.unique(labels)
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    ASC = np.array([])
    numC = np.array([])
    CoSeD = np.array([])
    viasc = 0
    if len(num_k) > 1:
        for i in num_k:
            data_of_cluster = X[labels == i]
            data_of_not_its = X[labels != i]
            isos = iso[labels == i]
            isos = (isos - min(isos)) / (max(isos) - min(isos))
            for j in range(len(data_of_cluster)):  # for each data of cluster j
                row = np.delete(data_of_cluster, j, 0)  # exclude the data j
                XX = data_of_cluster[j]
                a = closest_node(XX, row)
                b = closest_node(XX, data_of_not_its)
                ASC = np.hstack((ASC, ((b - a) / max(a, b)) * isos[j]))
            numC = np.hstack((numC, ASC.size))
            CoSeD = np.hstack((CoSeD, ASC.mean()))

        for k in range(len(numC)):
            viasc += numC[k] * CoSeD[k]

        viasc = viasc / sum(numC)
    else:
        viasc = float("nan")

    return viasc

