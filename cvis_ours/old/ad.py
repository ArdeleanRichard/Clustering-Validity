import numpy as np
from collections import defaultdict, deque
import heapq


# Find normalization?
class ArborisDistanceCalculator:
    """
    Arboris distance = distance of the tree
    MST-based distance computation
    """

    def __init__(self, data, n_neighbors=5):
        """
        Initialize with data and build MST.

        Parameters:
        - data: ndarray, shape (n_samples, n_features)
        - n_neighbors: int, number of nearest neighbors for graph construction
        """
        self.data = data
        self.n_samples = len(data)
        self.n_neighbors = n_neighbors

        # Build MST
        self.mst_edges = self._build_mst()
        self.adj = self._build_adj_list()

        # Cache for distance queries
        self._distance_cache = {}

    def _build_mst(self):
        """Build MST using Prim's with k-NN.

        Hot-path optimisation: instead of computing ||data - data[v]||^2 via a
        temporary (N, D) subtraction array each iteration, we reuse precomputed
        per-row squared norms and reduce to a single matrix-vector multiply:

            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b

        This avoids allocating an (N, D) intermediate on every step, which was
        the dominant cost in high-dimensional data.  Outputs are identical to
        the original for any given dataset.

        Heap pruning optimisation: each unvisited node tracks the best (lowest)
        distance seen so far from any visited node. When a heap entry is popped,
        if its distance is worse than the current best known for that node, it is
        a stale duplicate and is discarded immediately — before doing any work.
        This keeps the effective heap size closer to O(N) rather than O(N*k),
        reducing both memory pressure and the log factor on every push/pop.
        """
        n = self.n_samples
        visited = np.zeros(n, dtype=bool)
        edges = []
        pq = []

        # Precompute squared norms once — O(N*D) total, reused N times.
        norms_sq = np.einsum('ij,ij->i', self.data, self.data)  # shape (N,)

        # best_known[v] = smallest dist_sq to v seen from any visited node so far.
        # Initialised to inf; updated whenever a cheaper edge to v is discovered.
        best_known = np.full(n, np.inf)
        best_known[0] = 0.0

        def _dist_sq_from(v: int) -> np.ndarray:
            """Squared Euclidean distances from point v to all points."""
            d = norms_sq + norms_sq[v] - 2.0 * (self.data @ self.data[v])
            np.maximum(d, 0.0, out=d)  # guard tiny negatives from float rounding
            return d

        # Start from point 0
        visited[0] = True

        distances_sq = _dist_sq_from(0)
        neighbors = np.argpartition(distances_sq[1:], min(self.n_neighbors, n - 2))[:min(self.n_neighbors, n - 1)] + 1

        for neighbor in neighbors:
            d = float(distances_sq[neighbor])
            if d < best_known[neighbor]:
                best_known[neighbor] = d
                heapq.heappush(pq, (d, 0, int(neighbor)))

        while len(edges) < n - 1 and pq:
            dist_sq, u, v = heapq.heappop(pq)

            # Skip if already visited, or if a cheaper edge to v was found after
            # this entry was pushed (stale duplicate).
            if visited[v] or dist_sq > best_known[v]:
                continue

            edges.append((u, v, np.sqrt(dist_sq)))
            visited[v] = True

            unvisited_mask = ~visited
            if np.any(unvisited_mask):
                distances_sq = _dist_sq_from(v)
                distances_sq[visited] = np.inf

                k_actual = min(self.n_neighbors, int(np.sum(unvisited_mask)))
                if k_actual > 0:
                    neighbors = np.argpartition(distances_sq, k_actual - 1)[:k_actual]

                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            d = float(distances_sq[neighbor])
                            # Only push if this edge improves the best known
                            # distance to this neighbor, pruning stale entries.
                            if d < best_known[neighbor]:
                                best_known[neighbor] = d
                                heapq.heappush(pq, (d, v, int(neighbor)))

        return edges

    def _build_adj_list(self):
        """Build adjacency list from edges."""
        adj = defaultdict(list)
        for u, v, dist in self.mst_edges:
            adj[u].append((v, dist))
            adj[v].append((u, dist))
        return adj

    def get_distance(self, start, end):
        """
        Get MST path maximum edge distance between two points.
        Uses caching for repeated queries.
        """
        if start == end:
            return 0.0

        # Check cache (symmetric)
        cache_key = (min(start, end), max(start, end))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # BFS to find path
        queue = deque([start])
        parent = {start: None}
        parent_dist = {start: 0.0}

        found = False
        while queue:
            node = queue.popleft()

            if node == end:
                found = True
                break

            for neighbor, dist in self.adj[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    parent_dist[neighbor] = dist
                    queue.append(neighbor)

        if not found:
            raise ValueError(f"No path between {start} and {end}")

        # Find max edge by backtracking
        max_dist = 0.0
        current = end
        while parent[current] is not None:
            max_dist = max(max_dist, parent_dist[current])
            current = parent[current]

        # Cache result
        self._distance_cache[cache_key] = max_dist
        return max_dist

    def get_distances_to_point(self, target):
        """
        Distances from a target to all other points
        Returns array of shape (n_samples,).
        """
        distances = np.zeros(self.n_samples)

        # BFS from target
        queue = deque([target])
        parent = {target: None}
        parent_dist = {target: 0.0}
        max_dist_to = {target: 0.0}

        while queue:
            node = queue.popleft()

            for neighbor, edge_dist in self.adj[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    parent_dist[neighbor] = edge_dist
                    max_dist_to[neighbor] = max(max_dist_to[node], edge_dist)
                    distances[neighbor] = max_dist_to[neighbor]
                    queue.append(neighbor)

        return distances

    def get_distances_to_multiple(self, targets):
        """
        Distances from multiple targets to all points.
        Returns matrix of shape (len(targets), n_samples).
        """
        distance_matrix = np.zeros((len(targets), self.n_samples))
        for i, target in enumerate(targets):
            distance_matrix[i] = self.get_distances_to_point(target)
        return distance_matrix


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
    mean = data.mean(axis=0)                              # (D,)
    diff = data - mean                                    # (N, D)  — one allocation
    dist_to_mean = np.einsum('ij,ij->i', diff, diff)     # (N,)    — squared distances
    min_idx = int(np.argmin(dist_to_mean))

    return indices[min_idx] if indices is not None else min_idx

def _get_centroid_ids_from_data(data, labels, unique_labels):
    centroid_ids = np.array([
        _get_centroid_id_from_data(data[labels == label], indices=np.where(labels == label)[0])
        for label in unique_labels
    ])
    return centroid_ids