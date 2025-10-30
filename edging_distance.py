from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from constants import LABEL_COLOR_MAP


def centre_from_data(data):
    pairwise_distances = np.sum((data[:, np.newaxis] - data) ** 2, axis=-1)
    sum_squared_distances = np.sum(pairwise_distances, axis=1)
    min_index = np.argmin(sum_squared_distances)
    return data[min_index]


def k_nearest_neighbors(data, visited, query_point, n_neighbours=3):
    distances = np.sqrt(np.sum((data - query_point) ** 2, axis=1))
    sorted_distances = np.argsort(distances)
    nearest_indices = sorted_distances[~visited[sorted_distances]][:n_neighbours]

    return nearest_indices


def edging_distance(X, start, end, n_neighbours=5, lookahead=10, debug=False):
    start_id = np.where(np.all(X == start, axis=1))[0][0]
    end_id = np.where(np.all(X == end, axis=1))[0][0]

    if start_id == end_id:
        return 0

    path = []
    visited = np.zeros(len(X), dtype=bool)

    next_point_id = start_id
    next_point = np.copy(start)
    path.append(next_point)

    visited[next_point_id] = True

    saved_state = None
    la_count = 0
    looking_ahead_state = 0
    lookahead_counter = None

    lost_la = None
    while not next_point_id == end_id:
        if (len(visited) - np.count_nonzero(visited)) == 0:
            break

        if (len(visited) - np.count_nonzero(visited)) > n_neighbours:
            nearest_points_ids = k_nearest_neighbors(X, visited, next_point, n_neighbours)
        else:
            nearest_points_ids = np.where(visited == False)[0]

        distances_nearest_to_end = np.linalg.norm(X[nearest_points_ids] - end, axis=1 if len(X[nearest_points_ids].shape) > 1 else None)
        distance_current_to_end = np.linalg.norm(next_point - end)
        diff = distances_nearest_to_end - distance_current_to_end

        next_point_id = nearest_points_ids[np.argmin(distances_nearest_to_end)] if hasattr(distances_nearest_to_end, '__iter__') else nearest_points_ids
        if np.all(diff > 0):
            if looking_ahead_state == 0:
                looking_ahead_state = 1
                lookahead_counter = lookahead
                saved_state = [copy(path), copy(distance_current_to_end)]
                la_count +=1
        else:
            if saved_state is not None:
                _, old_distance_current_to_end = saved_state
                if distance_current_to_end < old_distance_current_to_end:
                    if looking_ahead_state == 1:
                        looking_ahead_state = 0

        if looking_ahead_state == 1:
            lookahead_counter = lookahead_counter - 1

            if lookahead_counter == 0:
                old_path, old_distance_current_to_end = saved_state

                if distance_current_to_end > old_distance_current_to_end:
                    lost_la = np.copy(path)
                    path = old_path

                next_point_id = end_id

        next_point = X[next_point_id]
        visited[nearest_points_ids] = True

        path.append(next_point)

    if len(path) == 1:
        return np.linalg.norm(path[0] - end)

    diff_vectors = np.diff(np.array(path), axis=0)
    distances = np.linalg.norm(diff_vectors, axis=1)
    max_edge = np.max(distances)

    if debug:
        label_color = [LABEL_COLOR_MAP[l] for l in labels]
        plt.title(f'Distance: {max_edge:.4f}')
        plt.scatter(X[:, 0], X[:, 1], color=label_color, marker='o', edgecolors='k')
        plt.scatter(cluster_means[:, 0], cluster_means[:, 1], color="white", marker="X", edgecolors='k', s=200)
        index = np.where((X == start).all(axis=1))[0][0]
        print(index)
        plt.scatter(path[0][0], path[0][1], color='white', label="start", edgecolors='k', s=100)
        plt.scatter(X[index,0], X[index, 1], color=label_color[index], label="start", edgecolors='k', s=100)
        plt.scatter(path[-1][0], path[-1][1], color='black', label="end", k='k', s=100)

        if lost_la is not None:
            for i in range(len(lost_la) - 1):
                plt.plot([lost_la[i][0], lost_la[i + 1][0]], [lost_la[i][1], lost_la[i + 1][1]], color='brown', linewidth=1)

        for i in range(len(path) - 1):
            plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='black', linewidth=2)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()
        plt.close()


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

