import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity
import warnings

warnings.filterwarnings("ignore")


def VIASCKDE(X, labels, kernel='gaussian', b_width=0.05):
    X = np.asarray(X)
    labels = np.asarray(labels)
    num_k = np.unique(labels)

    if len(num_k) <= 1:
        return float("nan")

    # Compute KDE once for all data
    kde = KernelDensity(kernel=kernel, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    total_weighted_score = 0.0
    total_count = 0

    num_samples, num_features = X.shape
    # Switch engine based on dimensionality
    USE_HIGH_DIM_BLAS = num_features > 20

    if USE_HIGH_DIM_BLAS:
        # --- PATHWAY A: HIGH-DIMENSIONAL (BLAS Accelerated Squared Distances) ---
        # Calculate squared distances up front. Sklearn's implementation is incredibly fast.
        # squared=True prevents wasting time taking millions of square roots.
        dist_matrix_sq = euclidean_distances(X, X, squared=True)

        for i in num_k:
            cluster_mask = labels == i
            if not np.any(cluster_mask): continue

            cluster_indices = np.where(cluster_mask)[0]
            other_indices = np.where(~cluster_mask)[0]

            isos = iso[cluster_indices]
            iso_min, iso_max = isos.min(), isos.max()
            isos = (isos - iso_min) / (iso_max - iso_min) if iso_max > iso_min else np.zeros_like(isos)

            # Within-cluster squared distances
            dist_within_sq = dist_matrix_sq[cluster_indices[:, None], cluster_indices]
            if dist_within_sq.shape[1] > 1:
                # O(N) selection for the second smallest squared distance (column 0 is self-distance 0.0)
                a_sq = np.partition(dist_within_sq, 1, axis=1)[:, 1]
            else:
                a_sq = np.zeros(dist_within_sq.shape[0])

            # Other-cluster squared distances
            if len(other_indices) > 0:
                b_sq = np.min(dist_matrix_sq[cluster_indices[:, None], other_indices], axis=1)
            else:
                b_sq = np.zeros(len(cluster_indices))

            # Take the square root ONLY on the final 1D vectors (huge time saver)
            a = np.sqrt(a_sq)
            b = np.sqrt(b_sq)

            max_ab = np.maximum(a, b)
            with np.errstate(divide='ignore', invalid='ignore'):
                ASC = np.where(max_ab > 0, ((b - a) / max_ab) * isos, 0.0)

            cluster_count = len(ASC)
            total_weighted_score += cluster_count * ASC.mean()
            total_count += cluster_count

    else:
        # --- PATHWAY B: LOW-DIMENSIONAL (KDTree) ---
        for i in num_k:
            cluster_mask = labels == i
            if not np.any(cluster_mask): continue

            cluster_indices = np.where(cluster_mask)[0]
            data_of_cluster = X[cluster_mask]
            data_of_not_its = X[~cluster_mask]

            isos = iso[cluster_mask]
            iso_min, iso_max = isos.min(), isos.max()
            isos = (isos - iso_min) / (iso_max - iso_min) if iso_max > iso_min else np.zeros_like(isos)

            kdtree_cluster = KDTree(data_of_cluster)
            dist_within, _ = kdtree_cluster.query(data_of_cluster, k=2)
            a = dist_within[:, 1] if dist_within.shape[1] > 1 else np.zeros(len(data_of_cluster))

            if len(data_of_not_its) > 0:
                kdtree_other = KDTree(data_of_not_its)
                b, _ = kdtree_other.query(data_of_cluster, k=1)
                b = b.ravel()
            else:
                b = np.zeros(len(data_of_cluster))

            max_ab = np.maximum(a, b)
            with np.errstate(divide='ignore', invalid='ignore'):
                ASC = np.where(max_ab > 0, ((b - a) / max_ab) * isos, 0.0)

            cluster_count = len(ASC)
            total_weighted_score += cluster_count * ASC.mean()
            total_count += cluster_count

    viasc = total_weighted_score / total_count
    return viasc



if __name__ == "__main__":
    import time
    from load_datasets import create_data5, create_data2, create_data3

    X, y = create_data5(1000, 4000)
    start = time.time()
    score = VIASCKDE(X, y)
    print(f"VIASCKDE: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data2(1000)
    start = time.time()
    score = VIASCKDE(X, y)
    print(f"VIASCKDE: {score:.3f} in {time.time()-start:.3f}s")

    X, y = create_data3(1000)
    start = time.time()
    score = VIASCKDE(X, y)
    print(f"VIASCKDE: {score:.3f} in {time.time()-start:.3f}s")