from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from constants import LABEL_COLOR_MAP


def centre_from_data(data):
    """
    Optimized: Uses einsum for faster pairwise distance calculation
    """
    # More efficient pairwise squared distance computation
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    sq_norms = np.sum(data ** 2, axis=1)
    dot_products = np.dot(data, data.T)
    pairwise_distances = sq_norms[:, np.newaxis] + sq_norms - 2 * dot_products

    sum_squared_distances = np.sum(pairwise_distances, axis=1)
    min_index = np.argmin(sum_squared_distances)
    return data[min_index]


def k_nearest_neighbors(data, visited, query_point, n_neighbours=3):
    """
    Optimized: Removed sqrt (unnecessary for sorting) and streamlined operations
    """
    distances_sq = np.sum((data - query_point) ** 2, axis=1)

    # Single pass filtering and sorting
    unvisited_mask = ~visited
    distances_sq[visited] = np.inf  # Mark visited as infinite distance

    # Get n smallest distances from unvisited points
    nearest_indices = np.argpartition(distances_sq, min(n_neighbours, np.sum(unvisited_mask) - 1))[:n_neighbours]

    # Filter out any visited that might have slipped through and sort by actual distance
    nearest_indices = nearest_indices[~visited[nearest_indices]]
    nearest_indices = nearest_indices[np.argsort(distances_sq[nearest_indices])][:n_neighbours]

    return nearest_indices


def edging_distance(X, start, end, n_neighbours=5, lookahead=10, debug=False):
    # Find indices of start and end
    start_id = np.where(np.all(X == start, axis=1))[0][0]
    end_id   = np.where(np.all(X == end,   axis=1))[0][0]

    if start_id == end_id:
        return 0

    n = len(X)

    path = []
    visited = np.zeros(n, dtype=bool)

    next_point_id = start_id
    next_point = np.copy(start)
    path.append(next_point)

    visited[next_point_id] = True

    visited_count = 1
    remaining = n - visited_count

    saved_state = None
    la_count = 0
    looking_ahead_state = 0
    lookahead_counter = None

    lost_la = None

    while next_point_id != end_id:

        # If no unvisited nodes remain, stop
        if remaining == 0:
            break

        # Select neighbors depending on how many remain
        if remaining > n_neighbours:
            nearest_points_ids = k_nearest_neighbors(X, visited, next_point, n_neighbours)
        else:
            nearest_points_ids = np.where(~visited)[0]

        # Compute distances
        distances_nearest_to_end = np.linalg.norm(
            X[nearest_points_ids] - end,
            axis=1 if len(X[nearest_points_ids].shape) > 1 else None
        )

        distance_current_to_end = np.linalg.norm(next_point - end)
        diff = distances_nearest_to_end - distance_current_to_end

        # Choose next point
        if hasattr(distances_nearest_to_end, "__iter__"):
            next_point_id = nearest_points_ids[np.argmin(distances_nearest_to_end)]
        else:
            next_point_id = nearest_points_ids

        # Lookahead logic
        if np.all(diff > 0):
            if looking_ahead_state == 0:
                looking_ahead_state = 1
                lookahead_counter = lookahead
                saved_state = [copy(path), copy(distance_current_to_end)]
                la_count += 1
        else:
            if saved_state is not None:
                _, old_dist = saved_state
                if distance_current_to_end < old_dist:
                    if looking_ahead_state == 1:
                        looking_ahead_state = 0

        if looking_ahead_state == 1:
            lookahead_counter -= 1
            if lookahead_counter == 0:
                old_path, old_dist = saved_state
                if distance_current_to_end > old_dist:
                    lost_la = np.copy(path)
                    path = old_path
                next_point_id = end_id

        # Advance to next point
        next_point = X[next_point_id]

        # Mark visited and update counters
        newly_visited = nearest_points_ids[~visited[nearest_points_ids]]
        visited[newly_visited] = True

        num_new = len(newly_visited)
        visited_count += num_new
        remaining = n - visited_count

        path.append(next_point)

    # If no path beyond start, return direct distance
    if len(path) == 1:
        return np.linalg.norm(path[0] - end)

    # Compute max edge
    diff_vectors = np.diff(np.array(path), axis=0)
    distances = np.linalg.norm(diff_vectors, axis=1)
    max_edge = np.max(distances)

    return max_edge






if __name__ == '__main__':
    # Example usage:
    # Create some points dataset
    points_dataset = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])

    # Define start and end points
    start_point = np.array((0, 0))
    end_point = np.array((4, 4))

    # Call the function
    result = edging_distance(points_dataset, start_point, end_point, debug=True)
    print("Max edge:", result)

