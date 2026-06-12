"""
Implementation of CDbw
https://github.com/alashkov83/CDbw

Citation:

"""

import importlib
import math
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def gen_dist_func(metric):
    """Obtain the distances function from scipy.spatial.distance package"""
    mod = importlib.import_module("scipy.spatial.distance")
    return getattr(mod, metric)


def filter_noise_lab(X, labels):
    """Filter noise points"""
    mask = labels != -1
    return labels[mask], X[mask]


def bind_noise_lab(X, labels, metric):
    """Bind noise points to nearest cluster"""
    labels = np.asarray(labels).copy()
    if -1 not in labels:
        return labels

    if len(set(labels)) == 1:
        raise ValueError('Labels contains noise point only')

    noise_mask = (labels == -1)
    valid_mask = ~noise_mask

    X_valid = X[valid_mask]
    X_noise = X[noise_mask]

    dists = cdist(X_valid, X_noise, metric=metric)
    nearest_valid_idx = np.argmin(dists, axis=0)

    labels[noise_mask] = labels[valid_mask][nearest_valid_idx]
    return labels


def comb_noise_lab(labels):
    """Combining all noise points into one cluster"""
    labels = np.asarray(labels).copy()
    labels[labels == -1] = np.max(labels) + 1
    return labels


def prep(X, labels):
    """Calculation necessary parameters"""
    dimension = X.shape[1]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    n_points_in_cl = np.zeros(n_clusters, dtype=int)
    std1_cl = np.zeros(n_clusters, dtype=float)

    coord_in_cl_list = []
    labels_in_cl_list = []

    for i, l in enumerate(unique_labels):
        mask = (labels == l)
        pts = X[mask]
        coord_in_cl_list.append(pts)
        labels_in_cl_list.append(np.where(mask)[0])

        n_points_in_cl[i] = len(pts)
        stdv = np.std(pts, axis=0)
        std1_cl[i] = math.sqrt(np.dot(stdv, stdv) / dimension)

    stdev = std1_cl[-1]  # Matches original logic quirk precisely

    n_max = max(n_points_in_cl)
    coord_in_cl = np.full((n_clusters, n_max, dimension), np.nan)
    labels_in_cl = np.full((n_clusters, n_max), -1)

    for i in range(n_clusters):
        n_pts = n_points_in_cl[i]
        coord_in_cl[i, :n_pts] = coord_in_cl_list[i]
        labels_in_cl[i, :n_pts] = labels_in_cl_list[i]

    return n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl, unique_labels


def rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl):
    """Select of representative points for each clusters"""
    mean_arr = np.zeros((n_clusters, dimension), dtype=float)
    n_rep = np.zeros(n_clusters, dtype=int)
    ch_vertices_list = []

    for i in range(n_clusters):
        n_pts = n_points_in_cl[i]
        pts = coord_in_cl[i, :n_pts]
        mean_arr[i] = np.mean(pts, axis=0)

        if n_pts >= 4:
            ch = ConvexHull(pts)
            vertices = ch.vertices
            n_rep[i] = vertices.size
            ch_vertices_list.append(vertices)
        else:
            n_rep[i] = n_pts
            ch_vertices_list.append(None)

    n_rep_max = np.max(n_rep)
    rep_in_cl = np.full((n_clusters, n_rep_max), -1)

    for i in range(n_clusters):
        n_r = n_rep[i]
        n_pts = n_points_in_cl[i]
        if n_pts >= 4:
            rep_in_cl[i, :n_r] = labels_in_cl[i, :n_pts][ch_vertices_list[i]]
        else:
            rep_in_cl[i, :n_r] = labels_in_cl[i, :n_pts]

    return mean_arr, n_rep, n_rep_max, rep_in_cl


def closest_rep(X, n_clusters, rep_in_cl, n_rep, metric, unique_pairs):
    """Select of the closest representative points for two clusters"""
    middle_point = defaultdict(list)
    dist_min = defaultdict(list)
    n_cl_rep = {}

    # Pre-extract representative coordinates to avoid indexing overhead in loops
    cluster_rep_nodes = [X[rep_in_cl[i, :n_rep[i]]] for i in range(n_clusters)]

    for i, j in unique_pairs:
        pts_i = cluster_rep_nodes[i]
        pts_j = cluster_rep_nodes[j]

        # Localized pair-wise matrix (small and extremely fast)
        dist_arr = cdist(pts_i, pts_j, metric=metric)

        r_mins = dist_arr.argmin(axis=1)
        c_mins = dist_arr.argmin(axis=0)

        rows = np.arange(n_rep[i])
        cols = r_mins
        valid = (c_mins[cols] == rows)

        u_indices = rows[valid]
        v_indices = cols[valid]

        n_cl_rep[(i, j)] = len(u_indices)

        if len(u_indices) > 0:
            m_pts = (pts_i[u_indices] + pts_j[v_indices]) / 2.0
            dist_min[(i, j)] = dist_arr[u_indices, v_indices].tolist()
            middle_point[(i, j)] = m_pts
        else:
            middle_point[(i, j)] = np.empty((0, X.shape[1]))
            dist_min[(i, j)] = []

    return middle_point, dist_min, n_cl_rep


def art_rep(X, n_clusters, rep_in_cl, n_rep, n_rep_max, mean_arr, s, dimension):
    """Calculate of the art representative points"""
    a_rep_shell = np.full((n_clusters, s, n_rep_max, dimension), np.nan)
    k_vals = (np.arange(s) / s)[:, np.newaxis, np.newaxis]

    for i in range(n_clusters):
        if n_rep[i] == 1:
            raise ValueError(f'Cluster No. {i:d} obtain only 1 point')
        n_r = n_rep[i]
        X_rep = X[rep_in_cl[i, :n_r]]
        a_rep_shell[i, :, :n_r, :] = (1.0 - k_vals) * X_rep[np.newaxis, :, :] + k_vals * mean_arr[i][np.newaxis, np.newaxis, :]

    return a_rep_shell


def compactness_optimized(X, labels, unique_labels, n_clusters, stdev, a_rep_shell, n_rep, n_points_in_cl, s, metric):
    """Clusters compactness evaluation using localized sub-matrices"""
    intra_dens_shell = np.zeros((n_clusters, s), dtype=float)

    for i in range(n_clusters):
        n_pts = n_points_in_cl[i]
        n_r = n_rep[i]

        # Isolate shell points for cluster i
        shell_pts = a_rep_shell[i, :s, :n_r, :].reshape(s * n_r, -1)

        # CRITICAL FIX: Only compute distance for points belonging to cluster i
        X_cluster = X[labels == unique_labels[i]]

        dists = cdist(X_cluster, shell_pts, metric=metric).reshape(n_pts, s, n_r)
        card = np.sum(dists < stdev, axis=(0, 2))

        intra_dens_shell[i, :] = card / (n_r * n_pts * stdev)

    intra_dens = np.sum(intra_dens_shell, axis=0) / n_clusters
    compact = np.sum(intra_dens) / s

    intra_change = np.sum(np.abs(np.diff(intra_dens))) / (s - 1)
    cohesion = compact / (1 + intra_change)
    return compact, cohesion


def separation_optimized(X, labels, unique_labels, n_clusters, stdev, middle_point, dist_min, n_cl_rep, n_points_in_cl, unique_pairs, metric):
    """Clusters separation evaluation using localized pair sub-matrices"""
    n_pairs = len(unique_pairs)
    dist_mm = np.zeros(n_pairs)
    dens1 = np.zeros(n_pairs)
    n_cl_arr = np.zeros(n_pairs)

    # Fast boolean masks lookup array
    cluster_masks = [labels == l for l in unique_labels]

    for p, (i, j) in enumerate(unique_pairs):
        m_pts = middle_point[(i, j)]
        n_rep_pair = n_cl_rep[(i, j)]

        n_cl_arr[p] = n_points_in_cl[i] + n_points_in_cl[j]

        if n_rep_pair > 0:
            # CRITICAL FIX: Only calculate distances for points inside cluster i and cluster j
            mask_pts = cluster_masks[i] | cluster_masks[j]
            X_pair = X[mask_pts]

            dist = cdist(X_pair, m_pts, metric=metric).T
            card = np.sum(dist < stdev, axis=1)

            d_min = np.array(dist_min[(i, j)])
            dist_mm[p] = np.sum(d_min) / n_rep_pair
            dens1[p] = np.sum(card * d_min)

    dist_mmm = np.zeros(n_clusters)
    nums = np.cumsum(np.arange(n_clusters))

    for i in range(n_clusters - 1):
        dist_mmm[i] = np.min(dist_mm[nums[i]:nums[i + 1]])

    dens_mean = dens1 / (stdev * n_cl_arr)
    inter_dens = np.sum(np.max(dens_mean)) / n_clusters
    dist_m = np.sum(dist_mmm) / n_clusters
    sep = dist_m / (1 + inter_dens)
    return sep


def CDbw(X, labels, metric="euclidean", alg_noise='comb', intra_dens_inf=False, s=3, multipliers=False):
    """Calculate CDbw-index for cluster validation"""
    if len(set(labels)) < 2 or len(set(labels)) > len(X) - 1:
        raise ValueError("No. of unique labels must be > 1 and < n_samples")
    if s < 2:
        raise ValueError("Parameter s must be > 2")
    elif alg_noise == 'bind':
        labels = bind_noise_lab(X, labels, metric=metric)
    elif alg_noise == 'comb':
        labels = comb_noise_lab(labels)
    elif alg_noise == 'filter':
        labels, X = filter_noise_lab(X, labels)

    labels = np.asarray(labels)

    n_clusters, stdev, dimension, n_points_in_cl, n_max, coord_in_cl, labels_in_cl, unique_labels = prep(X, labels)
    mean_arr, n_rep, n_rep_max, rep_in_cl = rep(n_clusters, dimension, n_points_in_cl, coord_in_cl, labels_in_cl)

    unique_pairs = []
    if n_clusters > 1:
        unique_pairs.append((1, 0))
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                unique_pairs.append((i, j))

    middle_point, dist_min, n_cl_rep = closest_rep(
        X, n_clusters, rep_in_cl, n_rep, metric, unique_pairs
    )

    try:
        a_rep_shell = art_rep(X, n_clusters, rep_in_cl, n_rep, n_rep_max, mean_arr, s, dimension)
    except ValueError:
        return 0

    compact, cohesion = compactness_optimized(
        X, labels, unique_labels, n_clusters, stdev, a_rep_shell, n_rep, n_points_in_cl, s, metric
    )

    if (np.isinf(compact) or np.isnan(compact)) and not intra_dens_inf:
        return 0

    sep = separation_optimized(
        X, labels, unique_labels, n_clusters, stdev, middle_point, dist_min, n_cl_rep,
        n_points_in_cl, unique_pairs, metric
    )

    cdbw = compact * cohesion * sep
    return (compact, cohesion, sep, cdbw) if multipliers else cdbw



if __name__ == "__main__":
    import time
    from load_datasets import create_data1, create_data2, create_data3

    X, y = create_data1(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data2(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data3(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time()-start:.3f}s")
