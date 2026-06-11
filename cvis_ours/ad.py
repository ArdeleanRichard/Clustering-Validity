import numpy as np
from collections import defaultdict, deque
import heapq


# Find normalization?
class ArborisDistanceCalculator:
    """
    Arboris distance = max-edge-weight (bottleneck) distance on the MST.

    Construction
    ------------
    _build_mst      – O(N·k·log N) Prim's with two optimisations:
                        1. Squared-norm reuse: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
                           avoids allocating an (N, D) intermediate each step.
                        2. Heap pruning via best_known[v]: stale heap entries are
                           discarded on pop, keeping heap size O(N) instead of O(N·k).
    _build_lifting  – O(N log N) binary lifting table, built only when N exceeds
                      the BFS crossover threshold. Stores ancestors and max edge
                      weights at every power-of-2 hop.

    Queries
    -------
    For N <= _LIFTING_THRESHOLD (default 2000):
        get_distances_to_point uses a plain BFS — O(N), tiny constant, no setup cost.
    For N > _LIFTING_THRESHOLD:
        get_distances_to_point uses _compute_row — O(N log N) but fully vectorised
        across all N nodes in parallel via numpy, faster than BFS at large N.
        Results are cached so repeated calls to the same target are O(N) copy.

    get_distance (scalar pair) always goes through get_distances_to_point and
    indexes the cached row, so it is O(N log N) first call, O(1) thereafter.
    """

    # Crossover point: below this BFS is faster; above it the vectorised
    # lifting query wins due to numpy parallelism over the LOG factor.
    _LIFTING_THRESHOLD = 2000

    def __init__(self, data, n_neighbors=5):
        """
        Parameters
        ----------
        data        : ndarray, shape (n_samples, n_features)
        n_neighbors : int, number of nearest neighbours for graph construction
        """
        self.data        = data
        self.n_samples   = len(data)
        self.n_neighbors = n_neighbors

        # Build MST
        self.mst_edges = self._build_mst()
        self.adj       = self._build_adj_list()

        # Build binary lifting table only when it will actually be used
        if self.n_samples > self._LIFTING_THRESHOLD:
            self._depth, self._up, self._up_w = self._build_lifting()
        else:
            self._depth = self._up = self._up_w = None

        # Row cache: target -> float64 array of shape (N,)
        self._distance_cache = {}

    # ------------------------------------------------------------------
    # MST construction
    # ------------------------------------------------------------------

    def _build_mst(self):
        """Prim's algorithm with k-NN candidate edges.

        Optimisation 1 — squared-norm reuse
        ------------------------------------
        Rather than computing ||data - data[v]||^2 by materialising an (N, D)
        difference array on every iteration, we precompute per-row squared norms
        once and reduce to a single matrix-vector multiply:

            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b

        Total cost O(N·D) for the precomputation, then O(N) per step instead of
        O(N·D) for the subtraction — the dot product is already O(N·D) and
        unavoidable, so no extra D-dimension work is added.

        Optimisation 2 — heap pruning via best_known
        --------------------------------------------
        best_known[v] tracks the cheapest dist_sq to unvisited node v seen so far.
        A heap entry is only pushed when it strictly improves best_known[v], and
        stale entries (dist_sq > best_known[v]) are discarded on pop. This keeps
        the live heap size O(N) instead of O(N·k), reducing both memory pressure
        and the log factor on every push/pop.
        """
        n    = self.n_samples
        data = self.data

        visited    = np.zeros(n, dtype=bool)
        best_known = np.full(n, np.inf)
        edges      = []
        pq         = []

        # Precompute squared norms once — O(N·D), reused every iteration.
        norms_sq = np.einsum('ij,ij->i', data, data)  # (N,)

        def _dist_sq_from(v: int) -> np.ndarray:
            """Squared Euclidean distances from point v to all points."""
            d = norms_sq + norms_sq[v] - 2.0 * (data @ data[v])
            np.maximum(d, 0.0, out=d)  # guard tiny negatives from float rounding
            return d

        # Seed from node 0
        visited[0]    = True
        best_known[0] = 0.0

        distances_sq = _dist_sq_from(0)
        k_init       = min(self.n_neighbors, n - 1)
        neighbors    = np.argpartition(distances_sq[1:], min(k_init - 1, n - 2))[:k_init] + 1

        for nb in neighbors:
            d = float(distances_sq[nb])
            if d < best_known[nb]:
                best_known[nb] = d
                heapq.heappush(pq, (d, 0, int(nb)))

        while len(edges) < n - 1 and pq:
            dist_sq, u, v = heapq.heappop(pq)

            # Skip already-visited nodes and stale heap entries.
            if visited[v] or dist_sq > best_known[v]:
                continue

            edges.append((u, v, np.sqrt(dist_sq)))
            visited[v] = True

            unvisited_mask = ~visited
            if not unvisited_mask.any():
                break

            distances_sq           = _dist_sq_from(v)
            distances_sq[visited]  = np.inf
            k_actual               = min(self.n_neighbors, int(unvisited_mask.sum()))

            if k_actual > 0:
                neighbors = np.argpartition(distances_sq, k_actual - 1)[:k_actual]
                for nb in neighbors:
                    if not visited[nb]:
                        d = float(distances_sq[nb])
                        if d < best_known[nb]:
                            best_known[nb] = d
                            heapq.heappush(pq, (d, v, int(nb)))

        return edges

    def _build_adj_list(self):
        """Build adjacency list from MST edges."""
        adj = defaultdict(list)
        for u, v, dist in self.mst_edges:
            adj[u].append((v, dist))
            adj[v].append((u, dist))
        return adj

    # ------------------------------------------------------------------
    # Binary lifting table  —  O(N log N), built only for large N
    # ------------------------------------------------------------------

    def _build_lifting(self):
        """Root the MST at node 0 and build binary lifting tables.

        up[k][v]   = 2^k-th ancestor of v.
        up_w[k][v] = max edge weight on the path going exactly 2^k hops up from v.

        Both arrays have shape (LOG, N) and are stored contiguously for cache-
        friendly column access during _compute_row.
        """
        N   = self.n_samples
        LOG = max(1, N.bit_length())  # ceil(log2 N) + 1 — enough for any path

        parent     = np.full(N, -1,  dtype=np.int32)
        par_weight = np.zeros(N,     dtype=np.float64)
        depth      = np.zeros(N,     dtype=np.int32)

        # Iterative DFS to avoid Python recursion limit on large trees.
        visited_dfs = np.zeros(N, dtype=bool)
        stack       = [0]
        iter_state  = [iter(self.adj[0])]
        visited_dfs[0] = True

        while stack:
            u = stack[-1]
            try:
                v, w = next(iter_state[-1])
                if not visited_dfs[v]:
                    visited_dfs[v] = True
                    parent[v]      = u
                    par_weight[v]  = w
                    depth[v]       = depth[u] + 1
                    stack.append(v)
                    iter_state.append(iter(self.adj[v]))
            except StopIteration:
                stack.pop()
                iter_state.pop()

        # Level 0: direct parent; root points to itself (sentinel).
        up    = np.empty((LOG, N), dtype=np.int32)
        up_w  = np.zeros((LOG, N), dtype=np.float64)
        mask  = parent >= 0
        up[0] = np.where(mask, parent, np.arange(N, dtype=np.int32))
        up_w[0] = par_weight  # par_weight[root] == 0 already

        # Fill higher levels: 2^k hop = two 2^(k-1) hops.
        for k in range(1, LOG):
            mid     = up[k - 1]           # (N,) ancestor indices
            up[k]   = up[k - 1][mid]
            up_w[k] = np.maximum(up_w[k - 1], up_w[k - 1][mid])

        return depth, up, up_w

    # ------------------------------------------------------------------
    # Vectorised row query  —  O(N log N) via numpy broadcasting
    # ------------------------------------------------------------------

    def _compute_row(self, target):
        """Bottleneck distance from *target* to every node, vectorised.

        Runs the standard binary-lifting LCA algorithm broadcast over all N
        nodes simultaneously using numpy arrays — no Python loop over nodes.
        Each of the LOG passes is a fully vectorised numpy operation.

        Returns float64 array of shape (N,).
        """
        N     = self.n_samples
        depth = self._depth
        up    = self._up    # (LOG, N)
        up_w  = self._up_w  # (LOG, N)
        LOG   = up.shape[0]

        # u_arr: "target" walker replicated for each of the N queries.
        # v_arr: each node is its own destination walker.
        u_arr = np.full(N, target, dtype=np.int32)
        v_arr = np.arange(N,       dtype=np.int32)
        max_u = np.zeros(N, dtype=np.float64)
        max_v = np.zeros(N, dtype=np.float64)

        dep_u = depth[u_arr]  # all equal depth[target]
        dep_v = depth[v_arr]

        # Canonicalise: u is always the deeper walker in each pair.
        swap        = dep_u < dep_v
        u_arr[swap], v_arr[swap] = v_arr[swap], u_arr[swap].copy()
        max_u[swap], max_v[swap] = max_v[swap], max_u[swap].copy()
        dep_u = depth[u_arr]
        dep_v = depth[v_arr]

        # Lift the deeper walker up until both sit at equal depth.
        diff = dep_u - dep_v
        for k in range(LOG):
            mask = ((diff >> k) & 1).astype(bool)
            if not mask.any():
                continue
            np.maximum(max_u, np.where(mask, up_w[k][u_arr], 0.0), out=max_u)
            u_arr = np.where(mask, up[k][u_arr], u_arr)

        # Nodes that are already equal after depth equalisation are at the LCA.
        same = u_arr == v_arr

        # Lift both walkers simultaneously until their next ancestor diverges.
        for k in range(LOG - 1, -1, -1):
            anc_u = up[k][u_arr]
            anc_v = up[k][v_arr]
            move  = (~same) & (anc_u != anc_v)
            if not move.any():
                continue
            np.maximum(max_u, np.where(move, up_w[k][u_arr], 0.0), out=max_u)
            np.maximum(max_v, np.where(move, up_w[k][v_arr], 0.0), out=max_v)
            u_arr = np.where(move, anc_u, u_arr)
            v_arr = np.where(move, anc_v, v_arr)

        # One final hop to the LCA for pairs that have not yet converged.
        not_same = ~same
        if not_same.any():
            np.maximum(max_u, np.where(not_same, up_w[0][u_arr], 0.0), out=max_u)
            np.maximum(max_v, np.where(not_same, up_w[0][v_arr], 0.0), out=max_v)

        result          = np.maximum(max_u, max_v)
        result[target]  = 0.0
        return result

    # ------------------------------------------------------------------
    # BFS row query  —  O(N), used for small N
    # ------------------------------------------------------------------

    def _bfs_row(self, target):
        """Bottleneck distances from *target* to all nodes via BFS.

        O(N) time and memory. Fastest for small N where numpy overhead
        in _compute_row outweighs its vectorisation benefit.

        Returns float64 array of shape (N,).
        """
        distances   = np.zeros(self.n_samples)
        queue       = deque([target])
        visited_bfs = {target: 0.0}  # node -> max edge so far on path from target

        while queue:
            node      = queue.popleft()
            node_dist = visited_bfs[node]

            for neighbor, edge_dist in self.adj[node]:
                if neighbor not in visited_bfs:
                    d                    = max(node_dist, edge_dist)
                    visited_bfs[neighbor] = d
                    distances[neighbor]  = d
                    queue.append(neighbor)

        return distances

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_distances_to_point(self, target):
        """Bottleneck distances from *target* to all N points.

        Returns array of shape (n_samples,).

        Uses BFS for N <= _LIFTING_THRESHOLD (faster due to lower constant),
        and the vectorised binary-lifting query for larger N.
        Results are cached: repeated calls for the same target are O(N) copy.
        """
        row = self._distance_cache.get(target)
        if row is None:
            if self._up is None:
                row = self._bfs_row(target)
            else:
                row = self._compute_row(target)
            self._distance_cache[target] = row
        return row.copy()

    def get_distances_to_multiple(self, targets):
        """Bottleneck distances from each target to all N points.

        Returns matrix of shape (len(targets), n_samples).
        """
        result = np.empty((len(targets), self.n_samples), dtype=np.float64)
        for i, t in enumerate(targets):
            result[i] = self.get_distances_to_point(t)
        return result

    def get_distance(self, start, end):
        """Bottleneck MST distance between two points.

        Computes the full row for *start* (cached), then indexes into it.
        O(N log N) or O(N) on first call per unique start; O(1) thereafter.
        """
        if start == end:
            return 0.0
        row = self._distance_cache.get(start)
        if row is None:
            row = self.get_distances_to_point(start)
            # get_distances_to_point already caches; retrieve without copy
            row = self._distance_cache[start]
        return float(row[end])


def _get_centroid_id_from_data(data, indices=None):
    if len(data) == 1:
        return indices[0] if indices is not None else 0

    # We want: argmin_i  sum_j ||x_i - x_j||^2
    # Expanding: sum_j ||x_i - x_j||^2
    #          = N*||x_i||^2 - 2*x_i·(N*mean) + sum_j||x_j||^2
    #          = N*(||x_i||^2 - 2*x_i·mean) + const
    #          = N*||x_i - mean||^2 + const   (up to a constant in ||mean||^2)
    #
    # So argmin is equivalent to finding the point closest to the centroid (mean).
    # This reduces cost from O(N^2 * D) to O(N * D) — no N×N matrix needed.
    mean          = data.mean(axis=0)                          # (D,)
    diff          = data - mean                                # (N, D) — one allocation
    dist_to_mean  = np.einsum('ij,ij->i', diff, diff)         # (N,)   — squared distances
    min_idx       = int(np.argmin(dist_to_mean))

    return indices[min_idx] if indices is not None else min_idx


def _get_centroid_ids_from_data(data, labels, unique_labels):
    centroid_ids = np.array([
        _get_centroid_id_from_data(data[labels == label], indices=np.where(labels == label)[0])
        for label in unique_labels
    ])
    return centroid_ids