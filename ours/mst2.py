import numpy as np
from collections import defaultdict, deque
import heapq


class MSTDistanceComputer:
    """
    Efficient MST-based distance computation with caching and optimizations.
    Build MST once, query distances many times.
    """

    def __init__(self, data, k=5, precompute_all=False):
        """
        Initialize with data and build MST once.

        Parameters:
        - data: ndarray, shape (n_samples, n_features)
        - k: int, number of nearest neighbors for MST construction
        - precompute_all: bool, whether to precompute all pairwise distances (memory vs speed tradeoff)
        """
        self.data = data
        self.n_samples = len(data)
        self.k = k

        # Build MST once
        self.edges = self._build_mst_optimized()
        self.adj = self._build_adjacency_list()

        # Cache for distance queries
        self._distance_cache = {}

        # Optionally precompute all distances (O(n²) space, O(1) query)
        if precompute_all and self.n_samples < 5000:  # Only for smaller datasets
            self._precompute_all_distances()

    def _build_mst_optimized(self):
        """Build MST using Prim's with k-NN, optimized version."""
        n = self.n_samples
        visited = np.zeros(n, dtype=bool)
        edges = []
        pq = []

        # Start from point 0
        visited[0] = True

        # Vectorized distance computation for initial neighbors
        distances_sq = np.sum((self.data - self.data[0]) ** 2, axis=1)
        neighbors = np.argpartition(distances_sq[1:], min(self.k, n - 2))[:min(self.k, n - 1)] + 1

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

                k_actual = min(self.k, np.sum(unvisited_mask))
                if k_actual > 0:
                    neighbors = np.argpartition(distances_sq, k_actual - 1)[:k_actual]

                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            heapq.heappush(pq, (distances_sq[neighbor], v, neighbor))

        return edges

    def _build_adjacency_list(self):
        """Build adjacency list from edges."""
        adj = defaultdict(list)
        for u, v, dist in self.edges:
            adj[u].append((v, dist))
            adj[v].append((u, dist))
        return adj

    def _precompute_all_distances(self):
        """Precompute all pairwise MST distances. Use for small datasets only."""
        print(f"Precomputing all {self.n_samples}² distances...")
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                dist = self.get_distance(i, j)
                # Both directions are cached by get_distance
        print("Precomputation complete!")

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
        Get distances from target to all other points efficiently.
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
        Efficiently compute distances from multiple targets to all points.
        Returns matrix of shape (len(targets), n_samples).
        """
        distance_matrix = np.zeros((len(targets), self.n_samples))
        for i, target in enumerate(targets):
            distance_matrix[i] = self.get_distances_to_point(target)
        return distance_matrix


def centroid_id_from_data_fast(data, indices=None):
    """Optimized centroid finding using vectorization."""
    if len(data) == 1:
        return indices[0] if indices is not None else 0

    # Compute all pairwise squared distances at once
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    pairwise_sq = np.sum(diff ** 2, axis=2)
    sum_sq = np.sum(pairwise_sq, axis=1)
    min_idx = np.argmin(sum_sq)

    return indices[min_idx] if indices is not None else min_idx


def mst_silhouette_score(data, labels, k=5):
    """Silhouette score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(data)

    if n_clusters == 1:
        return 0.0

    # Build MST once
    mst_computer = MSTDistanceComputer(data, k=k)

    # Find centroids
    centroid_ids = np.array([
        centroid_id_from_data_fast(data[labels == label],
                                   indices=np.where(labels == label)[0])
        for label in unique_labels
    ])

    # Get distances from all centroids to all points efficiently
    distance_matrix = mst_computer.get_distances_to_multiple(centroid_ids).T

    # Compute intra-cluster distances (distance to own cluster centroid)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    cluster_indices = np.array([label_to_idx[label] for label in labels])
    intra_distances = distance_matrix[np.arange(n_samples), cluster_indices]

    # Compute inter-cluster distances (minimum distance to other cluster centroids)
    # Set own cluster distance to inf
    distance_matrix_masked = distance_matrix.copy()
    distance_matrix_masked[np.arange(n_samples), cluster_indices] = np.inf
    inter_distances = np.min(distance_matrix_masked, axis=1)

    # Compute silhouette coefficients
    max_distances = np.maximum(intra_distances, inter_distances)
    silhouette_coefficients = np.where(
        max_distances > 0,
        (inter_distances - intra_distances) / max_distances,
        0.0
    )

    return np.mean(silhouette_coefficients)


def mst_davies_bouldin_score(data, labels, k=5):
    """Davies-Bouldin score using MST distances."""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST once
    mst_computer = MSTDistanceComputer(data, k=k)

    # Find centroids
    centroid_ids = np.array([
        centroid_id_from_data_fast(data[labels == label],
                                   indices=np.where(labels == label)[0])
        for label in unique_labels
    ])

    # Compute inter-cluster distances (between centroids)
    cluster_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = mst_computer.get_distance(centroid_ids[i], centroid_ids[j])
            cluster_distances[i, j] = dist
            cluster_distances[j, i] = dist

    # Compute intra-cluster scatter (mean distance to centroid)
    cluster_scatter = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) > 0:
            distances = mst_computer.get_distances_to_point(centroid_ids[i])
            cluster_scatter[i] = np.mean(distances[cluster_indices])

    # Compute Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_similarity = -np.inf
        for j in range(n_clusters):
            if i != j and cluster_distances[i, j] > 0:
                similarity = (cluster_scatter[i] + cluster_scatter[j]) / cluster_distances[i, j]
                max_similarity = max(max_similarity, similarity)
        if max_similarity > -np.inf:
            db_index += max_similarity

    return db_index / n_clusters


def mst_calinski_harabasz_score(data, labels, k=5):
    """Calinski-Harabasz score using MST distances."""
    n_samples = len(data)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Build MST once
    mst_computer = MSTDistanceComputer(data, k=k)

    # Find cluster centroids
    centroid_ids = np.array([
        centroid_id_from_data_fast(data[labels == label],
                                   indices=np.where(labels == label)[0])
        for label in unique_labels
    ])

    # Find overall centroid
    overall_centroid_id = centroid_id_from_data_fast(data)

    # Between-cluster sum of squares
    between_ss = 0.0
    for i, label in enumerate(unique_labels):
        cluster_size = np.sum(labels == label)
        dist = mst_computer.get_distance(overall_centroid_id, centroid_ids[i])
        between_ss += dist * cluster_size

    # Within-cluster sum of squares
    within_ss = 0.0
    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        distances = mst_computer.get_distances_to_point(centroid_ids[i])
        within_ss += np.sum(distances[cluster_indices])

    # Calinski-Harabasz index
    if within_ss == 0 or n_clusters == 1:
        return 0.0

    ch_index = (between_ss / (n_clusters - 1)) / (within_ss / (n_samples - n_clusters))
    return ch_index


def mst_separation_ratio(data, labels, k=5):
    """
    Optimized version of mst_idea: ratio of max intra-cluster to min inter-cluster distance.
    Lower is better (compact clusters, well-separated).
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # For within-cluster distances, build separate MSTs for each cluster
    max_intra_dist = 0.0

    for label in unique_labels:
        cluster_data = data[labels == label]
        if len(cluster_data) > 1:
            cluster_mst = MSTDistanceComputer(cluster_data, k=min(k, len(cluster_data) - 1))
            cluster_centroid_id = centroid_id_from_data_fast(cluster_data)

            # Maximum distance from centroid to any point in cluster
            distances = cluster_mst.get_distances_to_point(cluster_centroid_id)
            max_intra_dist = max(max_intra_dist, np.max(distances))

    # For inter-cluster distances, use full MST
    full_mst = MSTDistanceComputer(data, k=k)
    centroid_ids = np.array([
        centroid_id_from_data_fast(data[labels == label],
                                   indices=np.where(labels == label)[0])
        for label in unique_labels
    ])

    # Find minimum inter-cluster distance
    min_inter_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist = full_mst.get_distance(centroid_ids[i], centroid_ids[j])
            min_inter_dist = min(min_inter_dist, dist)

    if min_inter_dist == 0 or min_inter_dist == np.inf:
        return np.inf

    return max_intra_dist / min_inter_dist


# Performance comparison utility
def compare_performance():
    """Compare old vs new implementation performance."""
    from time import time
    from load_datasets import create_data4
    from sklearn.preprocessing import MinMaxScaler

    print("Performance Comparison")
    print("=" * 60)

    for n in [500, 1000, 2000]:
        print(f"\nDataset size: {n} samples")
        X, labels = create_data4(n)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        k = 5

        # Silhouette score
        start = time()
        score_opt = mst_silhouette_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Silhouette: {score_opt:.4f} in {time_opt:.3f}s")

        # Davies-Bouldin score
        start = time()
        score_opt = mst_davies_bouldin_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Davies-Bouldin: {score_opt:.4f} in {time_opt:.3f}s")

        # Calinski-Harabasz score
        start = time()
        score_opt = mst_calinski_harabasz_score(X, labels, k=k)
        time_opt = time() - start
        print(f"  Optimized Calinski-Harabasz: {score_opt:.4f} in {time_opt:.3f}s")

        # Separation ratio
        start = time()
        score_opt = mst_separation_ratio(X, labels, k=k)
        time_opt = time() - start
        print(f"  Separation Ratio: {score_opt:.4f} in {time_opt:.3f}s")


if __name__ == "__main__":
    compare_performance()