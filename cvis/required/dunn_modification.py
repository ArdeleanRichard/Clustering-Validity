import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import distance_matrix
from scipy.stats import entropy as calculate_entropy
from scipy.sparse import coo_matrix
from collections import Counter

def calculate_dunn_index(X=None, y_pred=None, use_modified=True, force_finite=True, finite_value=0.):
    centers, _ = compute_barycenters(X, y_pred)
    n_clusters = len(centers)
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Dunn index is undefined when y_pred has only 1 cluster.")
    # Calculate dmin
    dmin = np.inf
    if use_modified:
        for k0 in range(n_clusters - 1):
            for k1 in range(k0 + 1, n_clusters):
                points = X[y_pred == k1]
                if len(points) > 0:
                    dkk = np.min(cdist(points, centers[k0].reshape(1, -1), metric='euclidean'))
                    dmin = min(dmin, np.min(dkk))

    else:
        for kdx in range(n_clusters - 1):
            for k0 in range(kdx + 1, n_clusters):
                points1 = X[y_pred == kdx]
                points2 = X[y_pred == k0]
                if len(points1) > 0 and len(points2) > 0:
                    dkk = cdist(points1, points2, metric='euclidean')
                    dmin = min(dmin, np.min(dkk))

    # Calculate dmax
    dmax = 0.0
    for kdx in range(n_clusters):
        points = X[y_pred == kdx]
        if len(points) > 0:
            dk = np.max(pdist(points, metric="euclidean"))
            dmax = max(dmax, dk)
    return dmin / dmax