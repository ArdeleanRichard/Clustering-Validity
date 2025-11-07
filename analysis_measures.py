import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import pandas as pd


# ==================== DATASET IMBALANCE MEASURES ====================

def imbalance_ratio(y):
    """
    Calculate the Imbalance Ratio (IR) for binary or multi-class datasets.
    IR = majority_class_size / minority_class_size

    Parameters:
    -----------
    y : array-like
        Class labels

    Returns:
    --------
    float : Imbalance ratio
    dict : Class distribution
    """
    counts = Counter(y)
    max_count = max(counts.values())
    min_count = min(counts.values())

    ir = max_count / min_count

    return ir, dict(counts)


def multi_label_imbalance_metrics(y_multilabel):
    """
    Calculate imbalance metrics for multi-label datasets.

    Parameters:
    -----------
    y_multilabel : array-like, shape (n_samples, n_labels)
        Binary matrix where each column is a label

    Returns:
    --------
    dict : Dictionary containing IRLbl, MeanIR, MaxIR, and CVIR
    """
    n_samples, n_labels = y_multilabel.shape

    # Calculate imbalance ratio per label
    ir_per_label = []
    for i in range(n_labels):
        pos = np.sum(y_multilabel[:, i] == 1)
        neg = np.sum(y_multilabel[:, i] == 0)
        if pos > 0 and neg > 0:
            ir = max(pos, neg) / min(pos, neg)
            ir_per_label.append(ir)

    ir_per_label = np.array(ir_per_label)

    metrics = {
        'IRLbl': ir_per_label,  # Imbalance ratio per label
        'MeanIR': np.mean(ir_per_label),  # Mean imbalance ratio
        'MaxIR': np.max(ir_per_label),  # Maximum imbalance ratio
        'CVIR': np.std(ir_per_label) / np.mean(ir_per_label)  # Coefficient of variation
    }

    return metrics


# ==================== CLUSTER OVERLAP MEASURES ====================

def silhouette_coefficient(X, labels):
    """
    Calculate Silhouette Coefficient.
    Range: [-1, 1], where 0 indicates overlapping clusters.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like
        Cluster assignments

    Returns:
    --------
    float : Silhouette coefficient
    array : Per-sample silhouette scores
    """
    score = silhouette_score(X, labels)
    from sklearn.metrics import silhouette_samples
    sample_scores = silhouette_samples(X, labels)

    return score, sample_scores


def adjusted_rand_index(y_true, y_pred):
    """
    Calculate Adjusted Rand Index for comparing clustering with ground truth.
    Range: [-1, 1], where 1 indicates perfect agreement.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted cluster labels

    Returns:
    --------
    float : ARI score
    """
    return adjusted_rand_score(y_true, y_pred)


def normalized_mutual_information(y_true, y_pred):
    """
    Calculate Normalized Mutual Information.
    Range: [0, 1], where 1 indicates perfect agreement.

    Parameters:
    -----------
    y_true : array-like
        Ground truth labels
    y_pred : array-like
        Predicted cluster labels

    Returns:
    --------
    float : NMI score
    """
    return normalized_mutual_info_score(y_true, y_pred)


def overlap_rate_rvalue(X, labels):
    """
    Calculate R-value (overlap rate) between clusters.
    Uses distance-based approach to identify overlapping points.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like
        Cluster assignments

    Returns:
    --------
    float : R-value (proportion of overlapping points)
    dict : Pairwise overlap information
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate cluster centers
    centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

    # For each point, calculate distance to its cluster center and to nearest other center
    overlapping_points = 0
    total_points = len(X)

    for i, point in enumerate(X):
        own_label = labels[i]
        own_center_idx = np.where(unique_labels == own_label)[0][0]

        # Distance to own cluster center
        dist_to_own = np.linalg.norm(point - centers[own_center_idx])

        # Distance to nearest other cluster center
        other_centers = np.delete(centers, own_center_idx, axis=0)
        if len(other_centers) > 0:
            dist_to_nearest_other = np.min(cdist([point], other_centers))

            # Point is overlapping if it's closer to or similar distance to another cluster
            if dist_to_nearest_other <= dist_to_own * 1.2:  # 20% tolerance
                overlapping_points += 1

    r_value = overlapping_points / total_points

    return r_value


def bhattacharyya_coefficient(X, labels):
    """
    Calculate Bhattacharyya coefficient between cluster pairs.
    Assumes Gaussian distributions for each cluster.
    Range: [0, 1], where 0 = no overlap, 1 = complete overlap.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    labels : array-like
        Cluster assignments

    Returns:
    --------
    dict : Pairwise Bhattacharyya coefficients
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Fit Gaussian to each cluster
    distributions = {}
    for label in unique_labels:
        cluster_data = X[labels == label]
        mean = np.mean(cluster_data, axis=0)
        cov = np.cov(cluster_data.T)
        # Add small regularization to avoid singular matrices
        cov += np.eye(cov.shape[0]) * 1e-6
        distributions[label] = (mean, cov)

    # Calculate pairwise Bhattacharyya coefficients
    bc_matrix = {}
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1:]:
            mu1, sigma1 = distributions[label1]
            mu2, sigma2 = distributions[label2]

            # Bhattacharyya distance
            sigma = (sigma1 + sigma2) / 2

            try:
                term1 = 0.125 * (mu1 - mu2).T @ np.linalg.inv(sigma) @ (mu1 - mu2)
                term2 = 0.5 * np.log(np.linalg.det(sigma) /
                                     np.sqrt(np.linalg.det(sigma1) * np.linalg.det(sigma2)))

                bc_distance = term1 + term2
                bc_coefficient = np.exp(-bc_distance)

                bc_matrix[f"{label1}-{label2}"] = bc_coefficient
            except:
                bc_matrix[f"{label1}-{label2}"] = np.nan

    return bc_matrix


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Generate sample imbalanced dataset
    np.random.seed(42)

    # Example 1: Binary classification with imbalance
    y_binary = np.array([0] * 100 + [1] * 900)
    ir, dist = imbalance_ratio(y_binary)
    print("=== IMBALANCE MEASURES ===")
    print(f"Imbalance Ratio: {ir:.2f}")
    print(f"Class Distribution: {dist}")
    print()

    # Example 2: Multi-label dataset
    y_multilabel = np.random.randint(0, 2, size=(1000, 5))
    y_multilabel[:, 0] = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # Imbalanced label
    ml_metrics = multi_label_imbalance_metrics(y_multilabel)
    print("Multi-label Imbalance Metrics:")
    print(f"Mean IR: {ml_metrics['MeanIR']:.2f}")
    print(f"Max IR: {ml_metrics['MaxIR']:.2f}")
    print(f"CV IR: {ml_metrics['CVIR']:.2f}")
    print()

    # Example 3: Clustering with overlap
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                           cluster_std=1.5, random_state=42)

    print("=== CLUSTER OVERLAP MEASURES ===")

    # Silhouette Score
    sil_score, _ = silhouette_coefficient(X, y_true)
    print(f"Silhouette Coefficient: {sil_score:.3f}")

    # R-value
    r_val = overlap_rate_rvalue(X, y_true)
    print(f"R-value (Overlap Rate): {r_val:.3f} ({r_val * 100:.1f}% overlapping)")

    # Bhattacharyya Coefficient
    bc_scores = bhattacharyya_coefficient(X, y_true)
    print("Bhattacharyya Coefficients (pairwise):")
    for pair, score in bc_scores.items():
        print(f"  Clusters {pair}: {score:.3f}")

    # If you have ground truth and predictions
    y_pred = y_true.copy()
    y_pred[::10] = (y_pred[::10] + 1) % 3  # Add some noise

    print(f"\nAdjusted Rand Index: {adjusted_rand_index(y_true, y_pred):.3f}")
    print(f"Normalized Mutual Information: {normalized_mutual_information(y_true, y_pred):.3f}")