import time

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
    pairwise_distances = np.sum((data[:, np.newaxis] - data) ** 2, axis=-1)
    sum_squared_distances = np.sum(pairwise_distances, axis=1)
    min_index = np.argmin(sum_squared_distances)
    return indices[min_index] if indices is not None else min_index


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def k_nearest_neighbors(data, visited, query_point, n_neighbours=3):
    distances = np.sqrt(np.sum((data - query_point) ** 2, axis=1))
    sorted_distances = np.argsort(distances)
    nearest_indices = sorted_distances[~visited[sorted_distances]][:n_neighbours]
    return nearest_indices


def mst_prim_knn(data, k=5, start=0):
    """
    Build MST using Prim's algorithm with k-nearest neighbors.

    Args:
        data: array of data points
        k: number of nearest neighbors to consider
        start: index of the starting point (root of MST)

    Returns:
        list of (u, v, distance) tuples representing MST edges
    """
    n = len(data)
    visited = np.zeros(n, dtype=bool)
    edges = []
    pq = []

    # Start from the specified node
    visited[start] = True
    neighbors = k_nearest_neighbors(data, visited, data[start], k)
    for neighbor in neighbors:
        dist = euclidean_distance(data[start], data[neighbor])
        heapq.heappush(pq, (dist, start, neighbor))

    while len(edges) < n - 1 and pq:
        dist, u, v = heapq.heappop(pq)
        if visited[v]:
            continue

        edges.append((u, v, dist))
        visited[v] = True

        neighbors = k_nearest_neighbors(data, visited, data[v], k)
        for neighbor in neighbors:
            if not visited[neighbor]:
                d = euclidean_distance(data[v], data[neighbor])
                heapq.heappush(pq, (d, v, neighbor))

    return edges

def build_adjacency_list(edges, n_points):
    """
    Build adjacency list from MST edges.

    Args:
        edges: list of (point1_idx, point2_idx, distance) tuples
        n_points: total number of points

    Returns:
        adjacency list as dict: node -> [(neighbor, distance), ...]
    """
    adj = defaultdict(list)
    for u, v, dist in edges:
        adj[u].append((v, dist))
        adj[v].append((u, dist))
    return adj


def find_path_max_edge(edges, start, end):
    """
    Find the maximum edge distance along the path between two points in the MST.
    Uses BFS to find the path, then returns the max edge weight.

    Args:
        edges: list of (point1_idx, point2_idx, distance) tuples from MST
        start: starting point index
        end: ending point index

    Returns:
        max_distance: maximum edge distance along the path
        path: list of point indices from start to end

    Raises:
        ValueError: if no path exists between start and end
    """
    if start == end:
        return 0.0, np.array([start])

    # Build adjacency list
    adj = build_adjacency_list(edges, len(edges)+1)

    # BFS to find path
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        if node == end:
            # Found the path, now find max edge
            max_distance = 0.0
            for i in range(len(path) - 1):
                # Find edge distance between consecutive nodes in path
                curr, next_node = path[i], path[i + 1]
                for neighbor, dist in adj[curr]:
                    if neighbor == next_node:
                        max_distance = max(max_distance, dist)
                        break

            return max_distance, np.array(path)

        # Explore neighbors
        for neighbor, dist in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    raise ValueError(f"No path exists between point {start} and point {end}")


def max_edge_mst(data, start, end, k):
    edges = mst_prim_knn(data, k=k)
    max_dist, path = find_path_max_edge(edges, len(data), start, end)

    return max_dist


def plot_mst(data, mst_matrix):
    """Plot data points and MST edges."""
    plt.figure(figsize=(7, 7))
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=50, zorder=2, label='Points')

    # Plot MST edges
    n = len(data)
    for i in range(n):
        for j in range(n):
            if mst_matrix[i, j] > 0:
                p1, p2 = data[i], data[j]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1.5, alpha=0.8, zorder=1)

    plt.title("Minimum Spanning Tree (MST) over k-NN Graph")
    plt.legend()
    plt.show()


def plot_mst(data, edges):
    """Plot MST edges directly from edge list."""
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue', s=40, zorder=2, label='Points')

    for u, v, dist in edges:
        p1, p2 = data[u], data[v]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=1, alpha=0.7, zorder=1)

    plt.title("Minimum Spanning Tree (MST) over k-NN Graph")
    plt.legend()
    plt.show()



# Example usage with the MST functions from your code
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    # data = np.random.rand(100, 2) * 100
    data, _ = create_data1(1000)

    # Build MST
    print("Building MST...")
    edges = mst_prim_knn(data, k=10)
    print(f"MST has {len(edges)} edges")

    # Plot MST
    plot_mst(data, edges)

    # Test the path finding
    start_point = 0
    end_point = 50

    print(f"\nFinding path from point {start_point} to point {end_point}...")
    max_dist = find_path_max_edge(edges, len(data), start_point, end_point)

    print(f"\nPath from {start_point} to {end_point}") #of length {len(path)} nodes")
    # print(f"Path: {path[:10]}..." if len(path) > 10 else f"Path: {path}")
    print(f"Maximum edge distance in path: {max_dist:.4f}")

    # Test another pair
    start_point = 10
    end_point = 99
    max_dist = find_path_max_edge(edges, len(data), start_point, end_point)
    print(f"\nPath from {start_point} to {end_point}") #of length {len(path)} nodes")
    # print(f"Path: {path[:10]}..." if len(path) > 10 else f"Path: {path}")
    print(f"Maximum edge distance: {max_dist:.4f}")