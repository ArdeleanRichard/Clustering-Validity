def _var(
        X: np.ndarray,
        dist_kwargs=None,
) -> np.ndarray:
    """
    Helper function for the SD index, computing the "Var" vector. Optimized
    to eliminate the feature-by-feature loop for standard sqeuclidean metrics.
    """
    if dist_kwargs is None:
        dist_kwargs = {}
    else:
        dist_kwargs = dist_kwargs.copy()

    metric = dist_kwargs.setdefault("metric", "sqeuclidean")

    # OPTIMIZATION: If using standard sqeuclidean, bypass f_cdist and loop completely.
    # The mean of squared distances column-wise is exactly equivalent to variance.
    if metric == "sqeuclidean":
        if len(X.shape) == 2:
            # Axis 0 is the sample axis (N)
            return np.var(X, axis=0)
        elif len(X.shape) == 3:
            # Axis 0 is N, Axis 1 is w_t. We average over both to match the original behavior.
            return np.var(X, axis=(0, 1))

    # Fallback to original loop behavior if a custom metric is used
    center = np.expand_dims(compute_center(X), 0)
    if len(X.shape) == 2:
        Var = [
            np.mean(f_cdist(X[:, d:d + 1], center[:, d:d + 1], dist_kwargs))
            for d in range(X.shape[-1])
        ]
    elif len(X.shape) == 3:
        Var = [
            np.mean(f_cdist(X[:, :, d:d + 1], center[:, :, d:d + 1], dist_kwargs))
            for d in range(X.shape[-1])
        ]
    return np.array(Var)


def _dis(
        X: np.ndarray,
        clusters: List[List[int]],
        dist_kwargs={},
) -> float:
    """
    Helper function for the SD index, computing the "Dis" term.
    """
    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]
    d_btw_centroids = _dist_between_centroids(
        X, clusters=clusters, dist_kwargs=dist_kwargs
    )

    # For each center, compute the sum of distances to all other centers
    dis_aux = [
        np.sum(f_cdist(np.concatenate(centers, axis=0), c, dist_kwargs))
        for c in centers
    ]

    dis = float(
        (np.amax(d_btw_centroids) / np.amin(d_btw_centroids))
        * np.sum([1 / d_aux if d_aux != 0 else np.inf for d_aux in dis_aux])
    )

    return dis


def _scat(
        X: np.ndarray,
        clusters: List[List[int]],
        dist_kwargs={},
) -> float:
    """
    Helper function for the SD and SDbw indices.
    """
    N = len(X)
    k = len(clusters)
    total_var = np.linalg.norm(_var(X, dist_kwargs=dist_kwargs))

    scat = float(1 / k * np.sum([
        np.linalg.norm(_var(X[c], dist_kwargs=dist_kwargs)) / total_var
        for c in clusters
    ]))
    return scat


def SD_index(
        X: np.ndarray,
        clusters: List[List[int]],
        alpha: float = None,
        dist_kwargs={},
) -> float:
    """
    Compute the SD index for a given clustering.
    """
    scat = _scat(X, clusters=clusters, dist_kwargs=dist_kwargs)

    if alpha is None:
        alpha_aux = [
            np.sum(f_cdist(X, np.expand_dims(x, 0), dist_kwargs=dist_kwargs))
            for x in X
        ]

        d_intra = f_pdist(X, dist_kwargs=dist_kwargs)
        alpha = float(
            (np.amax(d_intra) / np.amin(d_intra))
            * np.sum(
                [1 / a_aux if a_aux != 0 else np.inf for a_aux in alpha_aux]
            )
        )
    dis = _dis(X, clusters=clusters, dist_kwargs=dist_kwargs)

    res = float(alpha * scat + dis)
    return res


def SDbw_index(
        X: np.ndarray,
        clusters: List[List[int]],
        dist_kwargs={},
) -> float:
    """
    Compute the SDbw index for a given clustering.
    """
    k = len(clusters)

    scat = _scat(X, clusters=clusters, dist_kwargs=dist_kwargs)

    centers = [np.expand_dims(compute_center(X[c]), 0) for c in clusters]

    nested_ijs = [
        [
            (i, j) for j in range(i + 1, k)
        ] for i in range(k - 1)
    ]

    ijs = sum(nested_ijs, [])

    u_ijs = [(centers[i] + centers[j]) / 2 for (i, j) in ijs]

    X_ijs = [
        np.concatenate((X[clusters[i]], X[clusters[j]]), axis=0)
        for (i, j) in ijs
    ]

    d_to_centroids = _dist_to_centroids(X, clusters, dist_kwargs=dist_kwargs)

    # Note: passed dist_kwargs={} implicitly to match original behavior
    # where _var(X[c]) was called without passing the outer dist_kwargs.
    stdev = float(1 / k * np.sqrt(np.sum([
        np.linalg.norm(_var(X[c]))
        for c in clusters
    ])))

    d_to_midpoints = [
        f_cdist(X_ij, u_ij, dist_kwargs=dist_kwargs)
        for X_ij, u_ij in zip(X_ijs, u_ijs)
    ]

    densities_ij = [
        np.sum(np.where(d_m <= stdev, np.ones_like(d_m), np.zeros_like(d_m)))
        for d_m in d_to_midpoints
    ]

    densities_i = [
        np.sum(np.where(d_i <= stdev, np.ones_like(d_i), np.zeros_like(d_i)))
        for d_i in d_to_centroids
    ]
    max_densities_ij = [
        np.amax([densities_i[i], densities_i[j]]) for (i, j) in ijs
    ]

    dens_bw = 1 / (k * (k - 1)) * 2 * np.sum([
        d_ij / max_d_ij if max_d_ij != 0 else np.inf
        for (d_ij, max_d_ij) in zip(densities_ij, max_densities_ij)
    ])

    res = float(scat + dens_bw)
    return res