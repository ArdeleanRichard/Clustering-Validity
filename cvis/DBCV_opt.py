"""
Optimized Implementation of Density-Based Clustering Validation "DBCV" (Higher is better)
https://github.com/FelSiq/DBCV

Citation:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.

OPTIMIZED VERSION - maintains identical results with improved performance

"""
"""
Optimized Implementation of Density-Based Clustering Validation "DBCV" (Higher is better)
https://github.com/FelSiq/DBCV

Citation:
Moulavi, Davoud, et al. "Density-based clustering validation."
Proceedings of the 2014 SIAM International Conference on Data Mining.
Society for Industrial and Applied Mathematics, 2014.

FULLY OPTIMIZED VERSION - maintains identical results with maximized performance
"""

import multiprocessing
import typing as t
import itertools
import functools

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import scipy.spatial.distance
import scipy.sparse.csgraph
import scipy.stats
import mpmath

_MP = mpmath.mp.clone()


def compute_pair_to_pair_dists(
        X: npt.NDArray[np.float64], metric: str
) -> npt.NDArray[np.float64]:
    """
    Compute the full pairwise dissimilarity matrix between rows of X using scipy.spatial.distance.cdist.
    - ensures diagonal is set to +inf (so a point is never considered its own nearest neighbor),
    - clips minimal distances to a small positive constant to avoid exact zeros.
    """
    dists = scipy.spatial.distance.cdist(X, X, metric=metric)
    # Avoid zeros (for stable inverse/power ops later)
    np.maximum(dists, 1e-12, out=dists)
    # NOTE: set self-distance to +inf to avoid self-neighbors.
    np.fill_diagonal(dists, val=np.inf)
    return dists


def _check_duplicated_samples(X: npt.NDArray[np.float64], threshold: float = 1e-9):
    """
    Detect near-duplicate rows in X using 1-NN distances.
    Raises ValueError if any pair distance < threshold.
    """
    if X.shape[0] <= 1:
        return

    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    dists, _ = nn.kneighbors(return_distance=True)

    if np.any(dists < threshold):
        raise ValueError("Duplicated samples have been found in X.")


def _convert_singleton_clusters_to_noise(
        y: npt.NDArray[np.int32], noise_id: int
) -> npt.NDArray[np.int32]:
    """
    Convert clusters with a single member to noise.
    """
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)
    singleton_clusters = cluster_ids[cluster_sizes == 1]

    if singleton_clusters.size == 0:
        return y

    return np.where(np.isin(y, singleton_clusters), noise_id, y)


def prim_mst(
        graph: npt.NDArray[np.float32], ind_root: int = 0
) -> npt.NDArray[np.float32]:
    """Python translation of the original implementation of Prim's MST in MATLAB.

    Reference source: https://github.com/pajaskowiak/dbcv/blob/main/src/MST_Edges.m
    """
    n = len(graph)
    intree = np.full(n, fill_value=False)  # nodes already in tree
    d = np.full(n, fill_value=np.inf)  # best distance to tree for each node

    d[ind_root] = 0
    v = ind_root

    # Pre-allocate arrays to store edges and weights (n-1 edges)
    node_inds = np.zeros((n - 1, 2), dtype=int)
    weights = np.zeros(n - 1, dtype=float)
    mst_parent = np.arange(n)

    for counter in range(n - 1):
        intree[v] = True

        # OPTIMIZATION: Vectorized calculation of best-known connection updates
        mask = ~intree
        graph_v = graph[v]
        update_mask = mask & (graph_v < d)
        d[update_mask] = graph_v[update_mask]
        mst_parent[update_mask] = v

        # OPTIMIZATION: Vectorized selection of the next vertex with the smallest distance
        remaining_indices = np.where(mask)[0]
        next_v = remaining_indices[np.argmin(d[remaining_indices])]

        node_inds[counter, :] = (mst_parent[next_v], next_v)
        weights[counter] = d[next_v]
        v = next_v

    inds_a, inds_b = node_inds.T

    mst = np.zeros_like(graph)
    mst[inds_a, inds_b] = weights
    mst[inds_b, inds_a] = weights

    return mst


def get_subarray(
        arr: npt.NDArray[np.float64],
        /,
        inds_a: t.Optional[npt.NDArray[np.int32]] = None,
        inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if inds_a is None:
        return arr
    if inds_b is None:
        # OPTIMIZATION: Chained 1D indexing is faster than multidimensional grid indexing
        return arr[inds_a][:, inds_a]
    return arr[inds_a][:, inds_b]


def compute_cluster_core_distance(
        dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool
) -> npt.NDArray[np.float64]:
    """
    Compute the core distance for each object in a cluster given the pairwise distances matrix dists.
    """
    n, _ = dists.shape
    orig_dists_dtype = dists.dtype

    if enable_dynamic_precision:
        dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

    core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1)

    if not enable_dynamic_precision:
        np.clip(core_dists, a_min=0.0, a_max=1e12, out=core_dists)

    np.power(core_dists, -1.0 / d, out=core_dists)

    if enable_dynamic_precision:
        core_dists = np.asarray(core_dists, dtype=orig_dists_dtype)

    return core_dists


def compute_mutual_reach_dists(
        dists: npt.NDArray[np.float64],
        d: float,
        enable_dynamic_precision: bool,
) -> npt.NDArray[np.float64]:
    """
    Compute core distances and mutual reachability distances for all pairs in a cluster.
    """
    core_dists = compute_cluster_core_distance(
        d=d, dists=dists, enable_dynamic_precision=enable_dynamic_precision
    )
    # OPTIMIZATION: Use in-place max updating to avoid intermediate array allocations
    mutual_reach_dists = np.maximum(dists, core_dists)
    np.maximum(mutual_reach_dists, core_dists.T, out=mutual_reach_dists)
    return (core_dists, mutual_reach_dists)


def get_internal_objects(
        mutual_reach_dists: npt.NDArray[np.float64], use_original_mst_implementation: bool
) -> npt.NDArray[np.float64]:
    if use_original_mst_implementation:
        mutual_reach_dists = np.copy(mutual_reach_dists)
        np.fill_diagonal(mutual_reach_dists, 0.0)
        mst = prim_mst(mutual_reach_dists)

        is_mst_edges = mst > 0.0
        degree = is_mst_edges.sum(axis=0)
        internal_node_inds = np.flatnonzero(degree > 1)

        if internal_node_inds.size == 0:
            return np.arange(mutual_reach_dists.shape[0]), mst

        internal_edge_weights = mst[np.ix_(internal_node_inds, internal_node_inds)]
        return internal_node_inds, internal_edge_weights

    else:
        # OPTIMIZATION: Calculate node degrees directly from the sparse graph structural layout
        mst_sparse = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
        out_degree = np.diff(mst_sparse.indptr)
        in_degree = np.bincount(mst_sparse.indices, minlength=mst_sparse.shape[0])
        degree = out_degree + in_degree

        internal_node_inds = np.flatnonzero(degree > 1)

        if internal_node_inds.size == 0:
            mst = mst_sparse.toarray()
            mst += mst.T
            return np.arange(mutual_reach_dists.shape[0]), mst

        # OPTIMIZATION: Construct only the required dense sub-graph slice instead of full N x N matrix inflation
        k = internal_node_inds.size
        internal_edge_weights = np.zeros((k, k), dtype=mutual_reach_dists.dtype)

        global_to_local = np.full(mst_sparse.shape[0], -1, dtype=np.int32)
        global_to_local[internal_node_inds] = np.arange(k, dtype=np.int32)

        r = np.repeat(np.arange(mst_sparse.shape[0]), np.diff(mst_sparse.indptr))
        c = mst_sparse.indices

        local_r = global_to_local[r]
        local_c = global_to_local[c]

        mask = (local_r >= 0) & (local_c >= 0)
        valid_r = local_r[mask]
        valid_c = local_c[mask]
        valid_data = mst_sparse.data[mask]

        internal_edge_weights[valid_r, valid_c] = valid_data
        internal_edge_weights[valid_c, valid_r] = valid_data

        return internal_node_inds, internal_edge_weights


def fn_density_sparseness(
        cls_inds: npt.NDArray[np.int32],
        dists: npt.NDArray[np.float64],
        d: int,
        enable_dynamic_precision: bool,
        use_original_mst_implementation: bool,
) -> t.Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """
    Compute Density Sparseness (DSC) and internal-core-distances for a single cluster.
    """
    (core_dists, mutual_reach_dists) = compute_mutual_reach_dists(
        dists=dists, d=d, enable_dynamic_precision=enable_dynamic_precision
    )
    (internal_node_inds, internal_edge_weights) = get_internal_objects(
        mutual_reach_dists,
        use_original_mst_implementation=use_original_mst_implementation,
    )
    dsc = float(internal_edge_weights.max()) if internal_edge_weights.size else 0.0
    internal_core_dists = core_dists[internal_node_inds]
    internal_node_inds = cls_inds[internal_node_inds]
    return (dsc, internal_core_dists, internal_node_inds)


def fn_density_separation(
        cls_i: int,
        cls_j: int,
        dists: npt.NDArray[np.float64],
        internal_core_dists_i: npt.NDArray[np.float64],
        internal_core_dists_j: npt.NDArray[np.float64],
) -> t.Tuple[int, int, float]:
    """
    Compute Density Separation (DSPC) between two clusters using their internal nodes and core distances.
    """
    # OPTIMIZATION: Use in-place updating to minimize allocations
    sep = np.maximum(dists, internal_core_dists_i)
    np.maximum(sep, internal_core_dists_j.T, out=sep)
    dspc_ij = float(sep.min()) if sep.size else np.inf
    return (cls_i, cls_j, dspc_ij)


def dbcv(
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.int32],
        metric: str = "sqeuclidean",
        noise_id: int = -1,
        check_duplicates: bool = True,
        n_processes: t.Union[int, str] = "auto",
        enable_dynamic_precision: bool = False,
        bits_of_precision: int = 512,
        use_original_mst_implementation: bool = False,
) -> float:
    """Compute DBCV metric."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)

    n, d = X.shape

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    y = _convert_singleton_clusters_to_noise(y, noise_id=noise_id)

    non_noise_inds = y != noise_id
    X = X[non_noise_inds, :]
    y = y[non_noise_inds]

    if y.size == 0:
        return 0.0

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

    if check_duplicates:
        _check_duplicated_samples(X)

    dists = compute_pair_to_pair_dists(X=X, metric=metric)

    # DSC: 'Density Sparseness of a Cluster'
    dscs = np.zeros(cluster_ids.size, dtype=float)

    # DSPC: 'Density Separation of a Pair of Clusters'
    min_dspcs = np.full(cluster_ids.size, fill_value=np.inf)

    internal_objects_per_cls: t.Dict[int, npt.NDArray[np.int32]] = {}
    internal_core_dists_per_cls: t.Dict[int, npt.NDArray[np.float32]] = {}

    # OPTIMIZATION: Group cluster indices in O(N log N) via stable sorting instead of repetitive scanning
    sort_idx = np.argsort(y, kind='stable')
    split_ptr = np.cumsum(cluster_sizes)[:-1]
    cls_inds = np.split(sort_idx, split_ptr)

    if n_processes == "auto":
        n_processes = 4 if y.size > 10000 else 1

    use_multiprocessing = n_processes > 1 and cluster_ids.size > 1

    fn_density_sparseness_ = functools.partial(
        fn_density_sparseness,
        d=d,
        enable_dynamic_precision=enable_dynamic_precision,
        use_original_mst_implementation=use_original_mst_implementation,
    )

    if use_multiprocessing:
        args = [(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]
        with _MP.workprec(bits_of_precision), multiprocessing.Pool(
                processes=min(n_processes, cluster_ids.size)
        ) as ppool:
            results = ppool.starmap(fn_density_sparseness_, args)
    else:
        # OPTIMIZATION: Avoid building giant lists in memory if sequential execution path is running
        with _MP.workprec(bits_of_precision):
            results = [fn_density_sparseness_(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]

    for cls_id, (dsc, internal_core_dists, internal_node_inds) in enumerate(results):
        internal_objects_per_cls[cls_id] = internal_node_inds
        internal_core_dists_per_cls[cls_id] = internal_core_dists
        dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    if n_cls_pairs > 0:
        use_multiprocessing_pairs = n_processes > 1 and n_cls_pairs > 1

        if use_multiprocessing_pairs:
            args = [
                (
                    cls_i,
                    cls_j,
                    get_subarray(
                        dists,
                        inds_a=internal_objects_per_cls[cls_i],
                        inds_b=internal_objects_per_cls[cls_j],
                    ),
                    internal_core_dists_per_cls[cls_i],
                    internal_core_dists_per_cls[cls_j],
                )
                for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
            ]
            with _MP.workprec(bits_of_precision), multiprocessing.Pool(
                    processes=min(n_processes, n_cls_pairs)
            ) as ppool:
                results = ppool.starmap(fn_density_separation, args)
        else:
            # OPTIMIZATION: Evaluate lazily to protect memory overhead from array slicing duplication
            with _MP.workprec(bits_of_precision):
                results = [
                    fn_density_separation(
                        cls_i,
                        cls_j,
                        get_subarray(
                            dists,
                            inds_a=internal_objects_per_cls[cls_i],
                            inds_b=internal_objects_per_cls[cls_j],
                        ),
                        internal_core_dists_per_cls[cls_i],
                        internal_core_dists_per_cls[cls_j],
                    )
                    for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
                ]

        for cls_i, cls_j, dspc_ij in results:
            min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
            min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False, posinf=1e12)
    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv_score = float(np.dot(vcs, cluster_sizes)) / n

    return dbcv_score



if __name__ == "__main__":
    import time
    from load_datasets import create_data1, create_data2, create_data3

    X, y = create_data1(1000)
    start = time.time()
    score = dbcv(X, y)
    print(f"DBCV: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data2(1000)
    start = time.time()
    score = dbcv(X, y)
    print(f"DBCV: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data3(1000)
    start = time.time()
    score = dbcv(X, y)
    print(f"DBCV: {score:.3f} in {time.time()-start:.3f}s")

