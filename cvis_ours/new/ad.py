import numpy as np
from collections import defaultdict, deque
import heapq


class ArborisDistanceCalculator:
    """
    Arboris distance = max-edge-weight distance on the MST.

    Strategy
    --------
    Init  – O(N·k·log(Nk)) Prim's MST  +  O(N log N) binary lifting table.
    Query – O(N log N) per *row* (all N distances from one target),
            fully vectorized across all N nodes in parallel.
            O(1) per scalar pair after the row is cached.

    This is optimal for the observed access pattern: a small number of
    centroid rows (k << N) are queried, not all N×N pairs.

    Binary lifting
    --------------
    up[k][v]   = 2^k-th ancestor of v in the MST rooted at node 0.
    up_w[k][v] = max edge weight on the path going exactly 2^k hops up from v.

    Vectorized row query
    --------------------
    To compute bottleneck(target, v) for ALL v simultaneously we run the
    standard O(log N) LCA algorithm but broadcast it across all N nodes at
    once using numpy, giving O(N log N) total work per row with a very
    small constant (pure numpy, no Python loops over nodes).
    """

    def __init__(self, data, n_neighbors=5):
        self.data        = data
        self.n_samples   = len(data)
        self.n_neighbors = n_neighbors

        self.mst_edges = self._build_mst()
        self.adj       = self._build_adj_list()

        # O(N log N): root the tree and build binary lifting tables as numpy arrays
        self._depth, self._up, self._up_w = self._build_lifting()

        # Row cache: target -> float64 array of shape (N,)
        self._distance_cache = {}

    # ------------------------------------------------------------------
    # MST construction  (identical to original)
    # ------------------------------------------------------------------

    def _build_mst(self):
        n       = self.n_samples
        visited = np.zeros(n, dtype=bool)
        edges   = []
        pq      = []

        visited[0] = True
        distances_sq = np.sum((self.data - self.data[0]) ** 2, axis=1)
        k = min(self.n_neighbors, n - 1)
        neighbors = np.argpartition(distances_sq[1:], min(k - 1, n - 2))[:k] + 1
        for nb in neighbors:
            heapq.heappush(pq, (distances_sq[nb], 0, nb))

        while len(edges) < n - 1 and pq:
            dist_sq, u, v = heapq.heappop(pq)
            if visited[v]:
                continue
            edges.append((u, v, float(np.sqrt(dist_sq))))
            visited[v] = True

            if (~visited).any():
                distances_sq = np.sum((self.data - self.data[v]) ** 2, axis=1)
                distances_sq[visited] = np.inf
                k_actual = min(self.n_neighbors, int((~visited).sum()))
                if k_actual > 0:
                    neighbors = np.argpartition(distances_sq, k_actual - 1)[:k_actual]
                    for nb in neighbors:
                        if not visited[nb]:
                            heapq.heappush(pq, (distances_sq[nb], v, nb))

        return edges

    def _build_adj_list(self):
        adj = defaultdict(list)
        for u, v, dist in self.mst_edges:
            adj[u].append((v, dist))
            adj[v].append((u, dist))
        return adj

    # ------------------------------------------------------------------
    # Binary lifting table  –  O(N log N), stored as contiguous numpy arrays
    # ------------------------------------------------------------------

    def _build_lifting(self):
        N   = self.n_samples
        LOG = max(1, N.bit_length())   # ceil(log2(N)) + 1

        parent     = np.full(N, -1, dtype=np.int32)
        par_weight = np.zeros(N, dtype=np.float64)
        depth      = np.zeros(N, dtype=np.int32)

        # Iterative DFS to root the tree at node 0
        visited = np.zeros(N, dtype=bool)
        # Use explicit stack with adj-list iterator state
        stack = [0]
        iter_state = [iter(self.adj[0])]
        visited[0] = True

        while stack:
            u = stack[-1]
            try:
                v, w = next(iter_state[-1])
                if not visited[v]:
                    visited[v]    = True
                    parent[v]     = u
                    par_weight[v] = w
                    depth[v]      = depth[u] + 1
                    stack.append(v)
                    iter_state.append(iter(self.adj[v]))
            except StopIteration:
                stack.pop()
                iter_state.pop()

        # Build lifting tables as 2-D numpy arrays: shape (LOG, N)
        up   = np.empty((LOG, N), dtype=np.int32)
        up_w = np.zeros((LOG, N), dtype=np.float64)

        # Level 0: direct parent (root points to itself)
        mask      = parent >= 0
        up[0]     = np.where(mask, parent, np.arange(N, dtype=np.int32))
        up_w[0]   = par_weight   # 0 for root (par_weight[0] == 0)

        for k in range(1, LOG):
            mid      = up[k-1]           # shape (N,)
            up[k]    = up[k-1][mid]
            up_w[k]  = np.maximum(up_w[k-1], up_w[k-1][mid])

        return depth, up, up_w

    # ------------------------------------------------------------------
    # Vectorized row query: bottleneck(target, ALL nodes) in O(N log N)
    # ------------------------------------------------------------------

    def _compute_row(self, target):
        """
        Return float64 array of shape (N,) where result[v] =
        bottleneck distance from target to v.

        Runs the standard binary-lifting LCA algorithm broadcast over all
        N nodes simultaneously — no Python loop over nodes.
        """
        N     = self.n_samples
        depth = self._depth
        up    = self._up      # (LOG, N)
        up_w  = self._up_w   # (LOG, N)
        LOG   = up.shape[0]

        # u_arr: the "target" walker replicated N times
        # v_arr: each node is its own walker
        u_arr = np.full(N, target, dtype=np.int32)
        v_arr = np.arange(N, dtype=np.int32)
        max_u = np.zeros(N, dtype=np.float64)
        max_v = np.zeros(N, dtype=np.float64)

        dep_u = depth[u_arr]   # all equal to depth[target]
        dep_v = depth[v_arr]

        # Swap so u is always the deeper node in each pair
        swap  = dep_u < dep_v
        u_arr[swap], v_arr[swap] = v_arr[swap], u_arr[swap].copy()
        max_u[swap], max_v[swap] = max_v[swap], max_u[swap].copy()
        dep_u = depth[u_arr]
        dep_v = depth[v_arr]

        # Lift the deeper walker up until both are at the same depth
        diff = dep_u - dep_v
        for k in range(LOG):
            mask = (diff >> k) & 1 == 1
            if not mask.any():
                continue
            np.maximum(max_u, up_w[k][u_arr] * mask, out=max_u)
            u_arr = np.where(mask, up[k][u_arr], u_arr)

        # Nodes already at the same node after depth equalisation
        same = u_arr == v_arr

        # Lift both walkers simultaneously until they diverge
        for k in range(LOG - 1, -1, -1):
            anc_u   = up[k][u_arr]
            anc_v   = up[k][v_arr]
            move    = (~same) & (anc_u != anc_v)
            if not move.any():
                continue
            np.maximum(max_u, up_w[k][u_arr] * move, out=max_u)
            np.maximum(max_v, up_w[k][v_arr] * move, out=max_v)
            u_arr = np.where(move, anc_u, u_arr)
            v_arr = np.where(move, anc_v, v_arr)

        # One final hop to the LCA for non-same pairs
        not_same = ~same
        if not_same.any():
            np.maximum(max_u, up_w[0][u_arr] * not_same, out=max_u)
            np.maximum(max_v, up_w[0][v_arr] * not_same, out=max_v)

        result = np.maximum(max_u, max_v)
        result[target] = 0.0
        return result

    # ------------------------------------------------------------------
    # Public API  (identical signatures and semantics to original)
    # ------------------------------------------------------------------

    def get_distances_to_point(self, target):
        """
        Bottleneck distances from *target* to all N points.
        Returns array of shape (n_samples,).
        First call O(N log N), cached thereafter O(N) copy.
        """
        row = self._distance_cache.get(target)
        if row is None:
            row = self._compute_row(target)
            self._distance_cache[target] = row
        return row.copy()

    def get_distances_to_multiple(self, targets):
        """
        Bottleneck distances from each target to all N points.
        Returns matrix of shape (len(targets), n_samples).
        """
        result = np.empty((len(targets), self.n_samples), dtype=np.float64)
        for i, t in enumerate(targets):
            result[i] = self.get_distances_to_point(t)
        return result

    def get_distance(self, start, end):
        """
        Bottleneck MST distance between two points.
        Computes the full row for `start` (cached), then indexes into it.
        O(N log N) first call per unique start, O(1) thereafter.
        """
        if start == end:
            return 0.0
        row = self._distance_cache.get(start)
        if row is None:
            row = self._compute_row(start)
            self._distance_cache[start] = row
        return float(row[end])


# ---------------------------------------------------------------------------
# Centroid functions
# ---------------------------------------------------------------------------

def _get_centroid_id_from_data(data, indices=None):
    if len(data) == 1:
        return indices[0] if indices is not None else 0
    diff    = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    sum_sq  = np.sum(np.sum(diff ** 2, axis=2), axis=1)
    min_idx = int(np.argmin(sum_sq))
    return indices[min_idx] if indices is not None else min_idx


def _get_centroid_ids_from_data(data, labels, unique_labels):
    return np.array([
        _get_centroid_id_from_data(
            data[labels == label],
            indices=np.where(labels == label)[0]
        )
        for label in unique_labels
    ])
