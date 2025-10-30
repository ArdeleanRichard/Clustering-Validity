import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import time

from load_datasets import create_data4
from mst_distance import centroid_id_from_data, find_path_max_edge, mst_prim_knn
from load_labelsets import diagonal_line, assign_labels_by_given_line, vertical_line, horizontal_line




def mst_silhouette_score(data, labels, k=5):
    """
    Calculate the Silhouette score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - db_score: float
        The Davies-Bouldin score.
    """
    n_samples, n_features = data.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0


    mst_edges = mst_prim_knn(data, k=k)

    centroid_ids = np.array([centroid_id_from_data(data[labels == i], indices=np.where(labels == i)[0]) for i in unique_labels])

    distance_matrix = np.zeros((n_samples, n_clusters))
    for sid, sample in enumerate(data):
        for cid, c_id in enumerate(centroid_ids):
            dist, path = find_path_max_edge(mst_edges, c_id, sid)
            distance_matrix[sid, cid] = dist

    # Compute the intra-cluster distances
    intra_cluster_distances = np.zeros(n_samples)
    for i in range(n_clusters):
        intra_cluster_distances[labels == unique_labels[i]] = np.mean(distance_matrix[labels == unique_labels[i], i])

    # Compute the inter-cluster distances for each sample
    inter_cluster_distances = np.min(distance_matrix + np.where(labels[:, np.newaxis] == unique_labels, np.inf, 0), axis=1)

    # Compute the silhouette coefficient for each sample
    silhouette_coefficients = (inter_cluster_distances - intra_cluster_distances) / np.maximum(inter_cluster_distances, intra_cluster_distances)

    # Compute the mean silhouette score over all samples
    mean_silhouette_score = np.mean(silhouette_coefficients)

    return mean_silhouette_score


def mst_davies_bouldin_score(data, labels, k=5):
    """
    Calculate the Davies-Bouldin score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - db_score: float
        The Davies-Bouldin score.
    """
    n_samples, n_features = data.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    mst_edges = mst_prim_knn(data, k=k)

    # Calculate cluster means
    centroid_ids = np.array([centroid_id_from_data(data[labels == i], indices=np.where(labels == i)[0]) for i in unique_labels])

    # Calculate cluster distances
    cluster_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                # cluster_distances[i, j] = np.linalg.norm(centroid_ids[i] - centroid_ids[j])
                # cluster_distances[i, j] = edging_distance(data, centroid_ids[i], centroid_ids[j], k, lookahead, debug)
                cluster_distances[i, j], path = find_path_max_edge(mst_edges, centroid_ids[i], centroid_ids[j])

    # Calculate cluster-wise scatter
    cluster_scatter = np.array([
        np.mean([
            # edging_distance(data, sample, centroid_ids[i], k, lookahead, debug)
            # for sid, sample in enumerate(data[labels == i])
            find_path_max_edge(mst_edges, centroid_ids[i], sid)[0]
            for sid, sample in zip(np.where(labels == i)[0], data[labels == i])
        ]) for i in range(n_clusters)
    ])

    # Calculate Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_similarity = -np.inf
        for j in range(n_clusters):
            if i != j:
                similarity = (cluster_scatter[i] + cluster_scatter[j]) / cluster_distances[i, j]
                if similarity > max_similarity:
                    max_similarity = similarity
        db_index += max_similarity
    db_index /= n_clusters

    return db_index


def mst_calinski_harabasz_score(data, labels, k=5):
    """
    Calculate the Calinski-Harabasz score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - ch_index: float
        The Calinski-Harabasz score.
    """
    n_samples, n_features = data.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    mst_edges = mst_prim_knn(data, k=k)

    # Calculate cluster means
    centroid_ids = np.array([centroid_id_from_data(data[labels == i], np.where(labels == i)[0]) for i in unique_labels])

    # Calculate overall mean
    centroid_overall_id = centroid_id_from_data(data)

    # Calculate between-cluster sum of squares
    # between_cluster_ss = np.sum([np.sum((centroid_ids[i] - overall_mean) ** 2) * np.sum(labels == i) for i in range(n_clusters)])
    between_cluster_ss = np.sum([
        # edging_distance(data, centroid_ids[i], overall_mean, k, lookahead, debug) * np.sum(labels == i)
        find_path_max_edge(mst_edges, centroid_overall_id, centroid_ids[i])[0] * np.sum(labels == i)
        for i in range(n_clusters)
    ])

    # Calculate within-cluster sum of squares
    # within_cluster_ss = np.sum([np.sum((X[labels == i] - centroid_ids[i]) ** 2) for i in range(n_clusters)])
    within_cluster_ss = np.sum([
            # edging_distance(data, sample, centroid_ids[i], k, lookahead, debug)
            # for i in range(n_clusters)
        # for sid, sample in enumerate(data[labels == i])
            find_path_max_edge(mst_edges, sid, centroid_ids[i])[0]
            for i in range(n_clusters)
        for sid, sample in zip(np.where(labels == i)[0], data[labels == i])
    ])

    # Calculate Calinski-Harabasz index
    ch_index = (between_cluster_ss / (n_clusters - 1)) / (within_cluster_ss / (n_samples - n_clusters))

    return ch_index



def mst_idea(data, labels, k=5):
    n_samples, n_features = data.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)



    # Calculate cluster means
    centroid_ids = np.array([centroid_id_from_data(data[labels == i], np.where(labels == i)[0]) for i in unique_labels])

    # within cluster distances
    all_dists = []
    all_dists_by_cluster = []
    for i in range(n_clusters):
        dists_by_cluster = []
        cluster = data[labels == i]
        cluster_centroid = centroid_id_from_data(cluster)
        mst_edges = mst_prim_knn(cluster, k=k)

        test = [dist for x,y,dist in mst_edges]
        print(np.max(np.array(test)))
        print(np.mean(np.array(test)))

        for sid, sample in enumerate(cluster):
            dist = find_path_max_edge(mst_edges, sid, cluster_centroid)[0]
            all_dists.append(dist)
            dists_by_cluster.append(dist)

        all_dists_by_cluster.append(dists_by_cluster)


    print([np.max(np.array(dists)) for dists in all_dists_by_cluster])
    print([np.mean(np.array(dists)) for dists in all_dists_by_cluster])

    all_dists = np.array(all_dists)
    print(np.max(all_dists))
    print(np.mean(all_dists))

    return all_dists


if __name__ == '__main__':
    n_samples = 1000
    X, gt = create_data4(n_samples)

    X = MinMaxScaler((-1, 1)).fit_transform(X)
    # X, gt = shuffle(X, gt, random_state=7)

    line = diagonal_line(X)
    dp = assign_labels_by_given_line(X, line)

    line = vertical_line(0)
    vl = assign_labels_by_given_line(X, line)

    line = horizontal_line(0)
    hl = assign_labels_by_given_line(X, line)

    k = 5
    la = 20

    # Set to vertical labelset test
    y = np.copy(gt)

    #
    # start = time.time()
    # score = silhouette_score(X, y)
    # print(f"SS: {score} in {time.time() - start:.2f}s")
    # start = time.time()
    # score = mst_silhouette_score(X, y, k=k)
    # print(f"ED-SS: {score} in {time.time() - start:.2f}s")
    # print()
    #
    #
    # start = time.time()
    # score = davies_bouldin_score(X, y)
    # print(f"DBS: {score} in {time.time() - start:.2f}s")
    # start = time.time()
    # score = mst_davies_bouldin_score(X, y, k=k)
    # print(f"ED-DBS: {score} in {time.time() - start:.2f}s")
    # print()
    #
    # start = time.time()
    # score = calinski_harabasz_score(X, y)
    # print(f"CHS: {score} in {time.time() - start:.2f}s")
    # start = time.time()
    # score = mst_calinski_harabasz_score(X, y, k=k)
    # print(f"ED-CHS: {score} in {time.time() - start:.2f}s")
    # print()

    score = mst_idea(X, y, k=k)


