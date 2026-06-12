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
    """Select of the closest representative points for two clusters via Global Matrix Slicing"""
    middle_point = defaultdict(list)
    dist_min = defaultdict(list)
    n_cl_rep = {}

    # Pack all cluster representative indices sequentially to calculate ONE global matrix
    cluster_rep_nodes = [rep_in_cl[i, :n_rep[i]] for i in range(n_clusters)]
    flat_reps = np.hstack(cluster_rep_nodes)
    offsets = np.cumsum([0] + [len(c) for c in cluster_rep_nodes])

    # Global Matrix 1: Multi-representative cross-distances
    D_reps = cdist(X[flat_reps], X[flat_reps], metric=metric)

    all_middle_points = []
    pair_mpt_slices = {}
    curr_mpt_idx = 0

    for i, j in unique_pairs:
        # Extract view slice out of global distance matrix without calculation cost
        dist_arr = D_reps[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]]

        r_mins = dist_arr.argmin(axis=1)
        c_mins = dist_arr.argmin(axis=0)

        rows = np.arange(n_rep[i])
        cols = r_mins
        valid = (c_mins[cols] == rows)

        u_indices = rows[valid]
        v_indices = cols[valid]

        n_cl_rep[(i, j)] = len(u_indices)

        if len(u_indices) > 0:
            pts_i = X[cluster_rep_nodes[i][u_indices]]
            pts_j = X[cluster_rep_nodes[j][v_indices]]

            m_pts = (pts_i + pts_j) / 2.0
            all_middle_points.append(m_pts)

            # Map distances directly from the matrix slice instead of calling distvec
            dist_min[(i, j)] = list(dist_arr[u_indices, v_indices])
            middle_point[(i, j)] = list(m_pts)

            pair_mpt_slices[(i, j)] = (curr_mpt_idx, curr_mpt_idx + len(u_indices))
            curr_mpt_idx += len(u_indices)
        else:
            pair_mpt_slices[(i, j)] = (curr_mpt_idx, curr_mpt_idx)

    return middle_point, dist_min, n_cl_rep, all_middle_points, pair_mpt_slices


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
    """Clusters compactness evaluation using Global Matrix Slicing"""
    intra_dens_shell = np.zeros((n_clusters, s), dtype=float)

    all_shell_pts = []
    shell_slices = []
    curr_shell_idx = 0

    for i in range(n_clusters):
        n_r = n_rep[i]
        shell_pts = a_rep_shell[i, :s, :n_r, :].reshape(s * n_r, -1)
        all_shell_pts.append(shell_pts)
        shell_slices.append((curr_shell_idx, curr_shell_idx + s * n_r))
        curr_shell_idx += s * n_r

    # Global Matrix 2: Mass point-to-shell distances
    D_X_shell = cdist(X, np.vstack(all_shell_pts), metric=metric)

    for i in range(n_clusters):
        n_pts = n_points_in_cl[i]
        n_r = n_rep[i]
        start, end = shell_slices[i]

        mask = (labels == unique_labels[i])
        dists = D_X_shell[mask, start:end].reshape(n_pts, s, n_r)
        card = np.sum(dists < stdev, axis=(0, 2))

        intra_dens_shell[i, :] = card / (n_r * n_pts * stdev)

    intra_dens = np.sum(intra_dens_shell, axis=0) / n_clusters
    compact = np.sum(intra_dens) / s

    intra_change = np.sum(np.abs(np.diff(intra_dens))) / (s - 1)
    cohesion = compact / (1 + intra_change)
    return compact, cohesion


def separation_optimized(X, labels, unique_labels, n_clusters, stdev, middle_point, dist_min, n_cl_rep, n_points_in_cl, all_middle_points, pair_mpt_slices, unique_pairs):
    """Clusters separation evaluation using Global Matrix Slicing"""
    n_pairs = len(unique_pairs)
    dist_mm = np.zeros(n_pairs)
    dens1 = np.zeros(n_pairs)
    n_cl_arr = np.zeros(n_pairs)

    if len(all_middle_points) > 0:
        # Global Matrix 3: Combined cluster point to boundary middle points
        D_X_middle = cdist(X, np.vstack(all_middle_points))
    else:
        D_X_middle = np.empty((len(X), 0))

    for p, (i, j) in enumerate(unique_pairs):
        d_min = np.array(dist_min[(i, j)])
        n_rep_pair = n_cl_rep[(i, j)]

        n_cl_arr[p] = n_points_in_cl[i] + n_points_in_cl[j]
        start_m, end_m = pair_mpt_slices[(i, j)]

        if n_rep_pair > 0:
            mask_pts = (labels == unique_labels[i]) | (labels == unique_labels[j])

            # Slice precomputed submatrix instead of running isolated cdist + np.vstack loops
            dist = D_X_middle[mask_pts, start_m:end_m].T
            card = np.sum(dist < stdev, axis=1)

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

    # Pre-generate the structured sequence of indexing pairs used by the original algorithm
    unique_pairs = []
    if n_clusters > 1:
        unique_pairs.append((1, 0))
    for i in range(2, n_clusters):
        for j in range(n_clusters):
            if i > j:
                unique_pairs.append((i, j))

    middle_point, dist_min, n_cl_rep, all_middle_points, pair_mpt_slices = closest_rep(
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
        n_points_in_cl, all_middle_points, pair_mpt_slices, unique_pairs
    )

    cdbw = compact * cohesion * sep
    return (compact, cohesion, sep, cdbw) if multipliers else cdbw


if __name__ == "__main__":
    import time
    from load_datasets import create_data1, create_data2, create_data3, create_data5

    X, y = create_data1(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time() - start:.3f}s")

    X, y = create_data2(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time() - start:.3f}s")

    X, y = create_data3(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time() - start:.3f}s")

    X, y = create_data5(1000)
    start = time.time()
    score = CDbw(X, y)
    print(f"CDbw: {score:.3f} in {time.time() - start:.3f}s")