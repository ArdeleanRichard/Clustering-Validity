import numpy as np
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
from load_datasets import create_data1


def centroid_id_from_data(data, indices=None):
    """
    Find the index (in the full dataset if indices provided) of the point
    minimizing the sum of squared distances to others in 'data'.
    """
    # Vectorized computation - more efficient than nested operations
    pairwise_distances = np.sum((data[:, np.newaxis] - data) ** 2, axis=-1)
    sum_squared_distances = np.sum(pairwise_distances, axis=1)
    min_index = np.argmin(sum_squared_distances)
    return indices[min_index] if indices is not None else min_index


def euclidean_distance_squared(p1, p2):
    """Compute squared Euclidean distance (avoid sqrt when possible)"""
    return np.sum((p1 - p2) ** 2)


def euclidean_distance(p1, p2):
    return np.sqrt(euclidean_distance_squared(p1, p2))


def k_nearest_neighbors(data, visited, query_point, n_neighbours=3):
    """Optimized k-NN using squared distances and partitioning"""
    # Use squared distances to avoid sqrt computation
    distances_sq = np.sum((data - query_point) ** 2, axis=1)

    # Create mask for unvisited points
    unvisited_mask = ~visited

    # Get indices of unvisited points
    unvisited_indices = np.where(unvisited_mask)[0]

    if len(unvisited_indices) == 0:
        return np.array([], dtype=int)

    # Get distances for unvisited points only
    unvisited_distances = distances_sq[unvisited_indices]

    # Use argpartition for O(n) time complexity instead of full sort O(n log n)
    k = min(n_neighbours, len(unvisited_indices))
    if k == len(unvisited_indices):
        return unvisited_indices

    partition_indices = np.argpartition(unvisited_distances, k - 1)[:k]
    nearest_indices = unvisited_indices[partition_indices]

    return nearest_indices


def build_mst(data, k=5, start=0):
    """
    Build MST using Prim's algorithm with k-nearest neighbors.
    Optimized with squared distances in heap operations.
    """
    n = len(data)
    visited = np.zeros(n, dtype=bool)
    edges = []
    pq = []

    # Start from the specified node
    visited[start] = True
    neighbors = k_nearest_neighbors(data, visited, data[start], k)

    # Use squared distances in heap for faster comparisons
    for neighbor in neighbors:
        dist_sq = euclidean_distance_squared(data[start], data[neighbor])
        heapq.heappush(pq, (dist_sq, start, neighbor))

    while len(edges) < n - 1 and pq:
        dist_sq, u, v = heapq.heappop(pq)

        if visited[v]:
            continue

        # Store actual distance in edges
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
    Build adjacency list from MST edges.
    """
    adj = defaultdict(list)
    for u, v, dist in edges:
        adj[u].append((v, dist))
        adj[v].append((u, dist))
    return adj


def find_path_max_edge(edges, start, end):
    """
    Find the maximum edge distance along the path between two points in the MST.
    Optimized to build adjacency list only once.
    """
    if start == end:
        return 0.0, np.array([start])

    # Build adjacency list (reuse if called multiple times by caching externally)
    adj = build_adjacency_list(edges)

    # BFS to find path with parent tracking for faster reconstruction
    queue = deque([start])
    visited = {start: None}  # Store parent instead of full path

    found = False
    while queue:
        node = queue.popleft()

        if node == end:
            found = True
            break

        # Explore neighbors
        for neighbor, dist in adj[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    if not found:
        raise ValueError(f"No path exists between point {start} and point {end}")

    # Reconstruct path from end to start using parent pointers
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = visited[current]
    path.reverse()

    # Find max edge distance along path
    max_distance = 0.0
    for i in range(len(path) - 1):
        curr, next_node = path[i], path[i + 1]
        # Direct lookup in adjacency list
        for neighbor, dist in adj[curr]:
            if neighbor == next_node:
                max_distance = max(max_distance, dist)
                break

    return max_distance, np.array(path)


def max_edge_mst(data, start, end, k):
    """Compute max edge in MST path"""
    edges = build_mst(data, k=k)
    max_dist, path = find_path_max_edge(edges, start, end)
    return max_dist


def plot_mst(data, edges):
    """Plot MST edges directly from edge list."""
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=40, zorder=2, label='Points')

    # Vectorize line plotting for better performance
    segments = []
    for u, v, dist in edges:
        segments.append([data[u], data[v]])

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors='black', linewidths=1, alpha=0.7, zorder=1)
    plt.gca().add_collection(lc)

    plt.title("Minimum Spanning Tree (MST) over k-NN Graph")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    data, _ = create_data1(1000)

    # Build MST
    print("Building MST...")
    edges = build_mst(data, k=10)
    print(f"MST has {len(edges)} edges")

    # Plot MST
    plot_mst(data, edges)

    # Test the path finding
    start_point = 0
    end_point = 50

    print(f"\nFinding path from point {start_point} to point {end_point}...")
    max_dist, path = find_path_max_edge(edges, start_point, end_point)

    print(f"\nPath from {start_point} to {end_point}")
    print(f"Maximum edge distance in path: {max_dist:.4f}")

    # Test another pair
    start_point = 10
    end_point = 99
    max_dist, path = find_path_max_edge(edges, start_point, end_point)
    print(f"\nPath from {start_point} to {end_point}")
    print(f"Maximum edge distance: {max_dist:.4f}")