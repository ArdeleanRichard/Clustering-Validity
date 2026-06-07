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
        """Build MST using Prim's with k-NN"""
        n = self.n_samples
        visited = np.zeros(n, dtype=bool)
        edges = []
        pq = []

        # Start from point 0
        visited[0] = True

        # Vectorized distance computation for initial neighbors
        distances_sq = np.sum((self.data - self.data[0]) ** 2, axis=1)
        neighbors = np.argpartition(distances_sq[1:], min(self.n_neighbors, n - 2))[:min(self.n_neighbors, n - 1)] + 1

        for neighbor in neighbors:
            heapq.heappush(pq, (distances_sq[neighbor], 0, neighbor))

        while len(edges) < n - 1 and pq:
            dist_sq, u, v = heapq.heappop(pq)

            if visited[v]:
                continue

            edges.append((u, v, np.sqrt(dist_sq)))
            visited[v] = True

            # Vectorized k-NN for new point
            unvisited_mask = ~visited
            if np.any(unvisited_mask):
                distances_sq = np.sum((self.data - self.data[v]) ** 2, axis=1)
                distances_sq[visited] = np.inf

                k_actual = min(self.n_neighbors, np.sum(unvisited_mask))
                if k_actual > 0:
                    neighbors = np.argpartition(distances_sq, k_actual - 1)[:k_actual]

                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            heapq.heappush(pq, (distances_sq[neighbor], v, neighbor))

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

    # Compute all pairwise squared distances at once
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    pairwise_sq = np.sum(diff ** 2, axis=2)
    sum_sq = np.sum(pairwise_sq, axis=1)
    min_idx = np.argmin(sum_sq)

    return indices[min_idx] if indices is not None else min_idx

def _get_centroid_ids_from_data(data, labels, unique_labels):
    centroid_ids = np.array([
        _get_centroid_id_from_data(data[labels == label], indices=np.where(labels == label)[0])
        for label in unique_labels
    ])
    return centroid_ids

