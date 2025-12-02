import numpy as np
from collections import defaultdict, deque
import heapq
from typing import Optional, Tuple, Dict, List


def centroid_id_from_data(data, indices=None):
    """
    Find the index of the point minimizing sum of squared distances.
    OPTIMIZED: Vectorized computation.
    """
    pairwise_distances = np.sum((data[:, np.newaxis] - data) ** 2, axis=-1)
    sum_squared_distances = np.sum(pairwise_distances, axis=1)
    min_index = np.argmin(sum_squared_distances)
    return indices[min_index] if indices is not None else min_index


def euclidean_distance_squared(p1, p2):
    """Compute squared Euclidean distance"""
    return np.sum((p1 - p2) ** 2)


def k_nearest_neighbors(data, visited, query_point, n_neighbours=3):
    """
    OPTIMIZED: Using squared distances and argpartition.
    """
    distances_sq = np.sum((data - query_point) ** 2, axis=1)
    unvisited_mask = ~visited
    unvisited_indices = np.flatnonzero(unvisited_mask)

    if unvisited_indices.size == 0:
        return np.array([], dtype=int)

    unvisited_distances = distances_sq[unvisited_indices]
    k = min(n_neighbours, unvisited_indices.size)

    if k == unvisited_indices.size:
        return unvisited_indices

    partition_indices = np.argpartition(unvisited_distances, k - 1)[:k]
    return unvisited_indices[partition_indices]


def build_mst(data, k=5, start=0):
    """
    Build MST using Prim's algorithm with k-nearest neighbors.
    OPTIMIZED: Reduced memory allocations and operations.
    """
    n = len(data)
    visited = np.zeros(n, dtype=bool)
    edges = []
    pq = []

    visited[start] = True
    neighbors = k_nearest_neighbors(data, visited, data[start], k)

    for neighbor in neighbors:
        dist_sq = euclidean_distance_squared(data[start], data[neighbor])
        heapq.heappush(pq, (dist_sq, start, neighbor))

    while len(edges) < n - 1 and pq:
        dist_sq, u, v = heapq.heappop(pq)

        if visited[v]:
            continue

        edges.append((u, v, np.sqrt(dist_sq)))
        visited[v] = True

        neighbors = k_nearest_neighbors(data, visited, data[v], k)
        for neighbor in neighbors:
            if not visited[neighbor]:
                d_sq = euclidean_distance_squared(data[v], data[neighbor])
                heapq.heappush(pq, (d_sq, v, neighbor))

    return edges


def build_adjacency_list(edges):
    """
    OPTIMIZED: Build adjacency list from MST edges.
    """
    adj = defaultdict(list)
    for u, v, dist in edges:
        adj[u].append((v, dist))
        adj[v].append((u, dist))
    return adj


# ============================================================================
# CRITICAL OPTIMIZATION: Cache-aware BFS with precomputed adjacency
# ============================================================================

def find_path_max_edge_fast(adj_list: Dict, start: int, end: int) -> Tuple[float, np.ndarray]:
    """
    OPTIMIZED: Find max edge using pre-built adjacency list.
    Uses early termination and efficient path reconstruction.
    """
    if start == end:
        return 0.0, np.array([start])

    queue = deque([start])
    visited = {start: None}
    found = False

    while queue:
        node = queue.popleft()
        if node == end:
            found = True
            break

        for neighbor, dist in adj_list[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    if not found:
        return np.inf, np.array([])

    # OPTIMIZATION: Reconstruct path and find max edge in single pass
    path = []
    max_distance = 0.0
    current = end

    while current is not None:
        path.append(current)
        parent = visited[current]

        if parent is not None:
            # Find edge weight during reconstruction
            for neighbor, dist in adj_list[current]:
                if neighbor == parent:
                    max_distance = max(max_distance, dist)
                    break

        current = parent

    path.reverse()
    return max_distance, np.array(path)


# ============================================================================
# BATCH OPTIMIZATIONS: Vectorized distance matrix computations
# ============================================================================

def compute_distance_matrix_batch(
        node_indices: np.ndarray,
        target_indices: np.ndarray,
        adj_list: Dict,
        cache: Optional[Dict] = None
) -> np.ndarray:
    """
    OPTIMIZED: Compute pairwise MST distances with optional caching.

    Uses symmetric matrix properties and caching to avoid redundant computations.
    """
    n_nodes = len(node_indices)
    n_targets = len(target_indices)

    # Check if we can use cache
    if cache is not None:
        use_cache = True
    else:
        cache = {}
        use_cache = False

    distance_matrix = np.zeros((n_nodes, n_targets))

    for i, node_idx in enumerate(node_indices):
        for j, target_idx in enumerate(target_indices):
            # Create canonical key (smaller, larger) for symmetric distances
            key = (min(node_idx, target_idx), max(node_idx, target_idx))

            if use_cache and key in cache:
                distance_matrix[i, j] = cache[key]
            else:
                dist, _ = find_path_max_edge_fast(adj_list, node_idx, target_idx)
                distance_matrix[i, j] = dist
                if use_cache:
                    cache[key] = dist

    return distance_matrix


def compute_centroid_distances_symmetric(
        centroid_indices: np.ndarray,
        adj_list: Dict
) -> np.ndarray:
    """
    OPTIMIZED: Compute pairwise centroid distances.
    Only computes upper triangle due to symmetry.
    """
    n_clusters = len(centroid_indices)
    distance_matrix = np.zeros((n_clusters, n_clusters))

    # Only compute upper triangle
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist, _ = find_path_max_edge_fast(adj_list, centroid_indices[i], centroid_indices[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def compute_intra_distances_vectorized(
        data: np.ndarray,
        labels: np.ndarray,
        centroid_indices: np.ndarray,
        adj_list: Dict,
        cluster_indices_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OPTIMIZED: Compute intra-cluster distances and scatter in single pass.

    Returns:
        mean_intra_distances: per-sample average distance to own centroid
        cluster_scatter: per-cluster average distance to centroid
    """
    n_samples = len(data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    mean_intra_distances = np.zeros(n_samples)
    cluster_scatter = np.zeros(n_clusters)

    for cluster_id, label in enumerate(unique_labels):
        cluster_indices = cluster_indices_list[cluster_id]
        centroid_idx = centroid_indices[cluster_id]

        if cluster_indices.size == 0:
            continue

        # Compute all distances in this cluster
        distances = np.array([
            find_path_max_edge_fast(adj_list, idx, centroid_idx)[0]
            for idx in cluster_indices
        ])

        avg_dist = np.mean(distances) if distances.size > 0 else 0.0
        cluster_scatter[cluster_id] = avg_dist
        mean_intra_distances[labels == label] = avg_dist

    return mean_intra_distances, cluster_scatter


# ============================================================================
# HIGH-LEVEL API: Pre-computation and shared state
# ============================================================================

class MSTDistanceCache:
    """
    OPTIMIZED: Cache MST structure and adjacency list for multiple queries.
    Principle from DBCV: build expensive structures once, reuse many times.
    """

    def __init__(self, data: np.ndarray, k: int = 5):
        self.data = data
        self.k = k
        self.mst_edges = None
        self.adj_list = None
        self._distance_cache = {}

    def build(self):
        """Build MST and adjacency list once."""
        if self.mst_edges is None:
            self.mst_edges = build_mst(self.data, k=self.k)
            self.adj_list = build_adjacency_list(self.mst_edges)
        return self

    def get_distance(self, start: int, end: int) -> float:
        """Get cached distance or compute and cache it."""
        key = (min(start, end), max(start, end))

        if key not in self._distance_cache:
            dist, _ = find_path_max_edge_fast(self.adj_list, start, end)
            self._distance_cache[key] = dist

        return self._distance_cache[key]

    def compute_distances_batch(
            self,
            node_indices: np.ndarray,
            target_indices: np.ndarray
    ) -> np.ndarray:
        """Compute batch distances with caching."""
        return compute_distance_matrix_batch(
            node_indices,
            target_indices,
            self.adj_list,
            cache=self._distance_cache
        )

    def clear_cache(self):
        """Clear distance cache to free memory."""
        self._distance_cache.clear()


def precompute_cluster_structure(
        data: np.ndarray,
        labels: np.ndarray,
        k: int = 5
) -> Tuple[np.ndarray, Dict, np.ndarray, List[np.ndarray]]:
    """
    OPTIMIZED: Precompute all shared structures needed for validity indices.
    This is the key principle from DBCV - compute expensive things once.

    Returns:
        centroid_indices: array of centroid indices per cluster
        adj_list: adjacency list for MST
        unique_labels: sorted unique label values
        cluster_indices_list: list of index arrays per cluster
    """
    unique_labels = np.unique(labels)

    # Build MST once
    mst_edges = build_mst(data, k=k)
    adj_list = build_adjacency_list(mst_edges)

    # Precompute cluster indices and centroids
    cluster_indices_list = [np.flatnonzero(labels == label) for label in unique_labels]

    centroid_indices = np.array([
        centroid_id_from_data(data[indices], indices=indices)
        for indices in cluster_indices_list
    ])

    return centroid_indices, adj_list, unique_labels, cluster_indices_list


# ============================================================================
# Legacy compatibility functions (for drop-in replacement)
# ============================================================================

def find_path_max_edge(edges, start, end):
    """
    Legacy API: Build adjacency list on-the-fly.
    For better performance, use precompute_cluster_structure instead.
    """
    adj_list = build_adjacency_list(edges)
    return find_path_max_edge_fast(adj_list, start, end)


def max_edge_mst(data, start, end, k):
    """Legacy API: Compute max edge in MST path"""
    edges = build_mst(data, k=k)
    max_dist, path = find_path_max_edge(edges, start, end)
    return max_dist


import numpy as np
import functools
import multiprocessing
from typing import Tuple, Optional


# ============================================================================
# CORE PRINCIPLE FROM DBCV: Precompute expensive structures, reuse everywhere
# ============================================================================

def mst_silhouette_score3(data, labels, k=5):
    """
    HIGHLY OPTIMIZED: MST-based Silhouette score.

    Key optimizations from DBCV:
    - Single MST build with precomputed adjacency list
    - Vectorized distance computations
    - Distance caching for symmetric queries
    - Early returns for edge cases
    """
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # OPTIMIZATION: Early return
    if n_clusters == 1:
        return 0.0

    # OPTIMIZATION: Precompute all shared structures once (DBCV principle)
    centroid_indices, adj_list, unique_labels, cluster_indices_list = \
        precompute_cluster_structure(data, labels, k)

    # OPTIMIZATION: Batch compute distances with caching
    all_indices = np.arange(n_samples)
    distance_cache = {}
    distance_matrix = compute_distance_matrix_batch(
        all_indices, centroid_indices, adj_list, cache=distance_cache
    )

    # OPTIMIZATION: Vectorized label-to-cluster mapping
    label_to_cluster_id = {label: i for i, label in enumerate(unique_labels)}
    cluster_ids = np.array([label_to_cluster_id[label] for label in labels])

    # Intra-cluster distances (to own centroid)
    intra_cluster_distances = distance_matrix[np.arange(n_samples), cluster_ids]

    # OPTIMIZATION: Vectorized per-cluster averaging
    intra_avg = np.zeros(n_samples)
    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        intra_avg[cluster_mask] = np.mean(intra_cluster_distances[cluster_mask])

    # Inter-cluster distances (min distance to other centroids)
    distance_matrix_masked = distance_matrix.copy()
    distance_matrix_masked[np.arange(n_samples), cluster_ids] = np.inf
    inter_cluster_distances = np.min(distance_matrix_masked, axis=1)

    # OPTIMIZATION: Vectorized silhouette coefficient calculation
    denominator = np.maximum(inter_cluster_distances, intra_avg)
    silhouette_coefficients = np.where(
        denominator > 0,
        (inter_cluster_distances - intra_avg) / denominator,
        0
    )

    return float(np.mean(silhouette_coefficients))


def mst_davies_bouldin_score3(data, labels, k=5):
    """
    HIGHLY OPTIMIZED: MST-based Davies-Bouldin score.

    Key optimizations from DBCV:
    - Precompute structures once
    - Symmetric distance matrix (only upper triangle)
    - Combined intra-distance computation
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # OPTIMIZATION: Early return
    if n_clusters == 1:
        return 0.0

    # OPTIMIZATION: Precompute all shared structures (DBCV principle)
    centroid_indices, adj_list, unique_labels, cluster_indices_list = \
        precompute_cluster_structure(data, labels, k)

    # OPTIMIZATION: Compute symmetric centroid distances (upper triangle only)
    cluster_distances = compute_centroid_distances_symmetric(centroid_indices, adj_list)

    # OPTIMIZATION: Compute cluster scatter in single pass
    _, cluster_scatter = compute_intra_distances_vectorized(
        data, labels, centroid_indices, adj_list, cluster_indices_list
    )

    # OPTIMIZATION: Vectorized Davies-Bouldin calculation
    db_index = 0.0
    for i in range(n_clusters):
        # Avoid division by zero
        valid_mask = (cluster_distances[i, :] > 0) & (np.arange(n_clusters) != i)

        if np.any(valid_mask):
            similarities = np.zeros(n_clusters)
            similarities[valid_mask] = (
                    (cluster_scatter[i] + cluster_scatter[valid_mask]) /
                    cluster_distances[i, valid_mask]
            )
            max_similarity = np.max(similarities)
        else:
            max_similarity = 0.0

        db_index += max_similarity

    return float(db_index / n_clusters) if n_clusters > 0 else 0.0


def mst_calinski_harabasz_score3(data, labels, k=5):
    """
    HIGHLY OPTIMIZED: MST-based Calinski-Harabasz score.

    Key optimizations from DBCV:
    - Single-pass variance computations
    - Cached distance lookups
    - Vectorized summations
    """
    n_samples = data.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # OPTIMIZATION: Early return
    if n_clusters == 1 or n_samples == n_clusters:
        return 0.0

    # OPTIMIZATION: Precompute all shared structures
    centroid_indices, adj_list, unique_labels, cluster_indices_list = \
        precompute_cluster_structure(data, labels, k)

    # Overall centroid
    overall_centroid_id = centroid_id_from_data(data)

    # OPTIMIZATION: Vectorized between-cluster variance
    between_cluster_ss = 0.0
    cluster_sizes = np.array([len(indices) for indices in cluster_indices_list])

    for i, centroid_idx in enumerate(centroid_indices):
        dist, _ = find_path_max_edge_fast(adj_list, overall_centroid_id, centroid_idx)
        between_cluster_ss += dist * cluster_sizes[i]

    # OPTIMIZATION: Single-pass within-cluster variance
    within_cluster_ss = 0.0
    for i, indices in enumerate(cluster_indices_list):
        for idx in indices:
            dist, _ = find_path_max_edge_fast(adj_list, idx, centroid_indices[i])
            within_cluster_ss += dist

    # OPTIMIZATION: Safe division
    if within_cluster_ss == 0:
        return 0.0

    ch_index = (between_cluster_ss / (n_clusters - 1)) / (within_cluster_ss / (n_samples - n_clusters))
    return float(ch_index)
