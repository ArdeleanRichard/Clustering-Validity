import numpy as np
from scipy.special import comb

from load_datasets import create_unbalance


def adjusted_rand_index(labels_true, labels_pred):
    """
    Calculate the Adjusted Rand Index (ARI) between two clusterings.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels

    Returns:
    --------
    ari : float
        Adjusted Rand Index. Range: [-1, 1]
        1.0 = perfect match
        0.0 = random labeling
        negative values = worse than random
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # Check inputs
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("labels_true and labels_pred must have same length")

    # Get unique labels
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Build contingency table
    n = len(labels_true)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    # Sum of combinations for each row and column
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency, axis=0))

    # Sum of combinations for each cell in contingency table
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())

    # Total combinations
    total_comb = comb(n, 2)

    # Expected index (for adjustment)
    expected_index = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0

    # Max index
    max_index = (sum_comb_c + sum_comb_k) / 2

    # Adjusted Rand Index
    if max_index - expected_index == 0:
        return 1.0 if sum_comb == expected_index else 0.0

    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari


# Pure NumPy implementation (faster for large datasets)
def adjusted_rand_index_numpy(labels_true, labels_pred):
    """
    Optimized NumPy implementation of Adjusted Rand Index.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels

    Returns:
    --------
    ari : float
        Adjusted Rand Index
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    n = len(labels_true)

    # Create contingency table using broadcasting
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)

    # Build contingency matrix efficiently
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    np.add.at(contingency, (class_idx, cluster_idx), 1)

    # Calculate sums
    sum_rows = np.sum(contingency, axis=1)
    sum_cols = np.sum(contingency, axis=0)

    # Combination calculations: C(n,2) = n*(n-1)/2
    def comb2(x):
        """Calculate C(x, 2) = x * (x-1) / 2"""
        return x * (x - 1) / 2

    sum_comb_c = np.sum(comb2(sum_rows))
    sum_comb_k = np.sum(comb2(sum_cols))
    sum_comb = np.sum(comb2(contingency))
    total_comb = comb2(n)

    # Expected and max indices
    expected_index = sum_comb_c * sum_comb_k / total_comb if total_comb > 0 else 0
    max_index = (sum_comb_c + sum_comb_k) / 2

    # ARI calculation
    if abs(max_index - expected_index) < 1e-10:
        return 1.0 if abs(sum_comb - expected_index) < 1e-10 else 0.0

    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari


def balanced_adjusted_rand_index(labels_true, labels_pred, method='macro'):
    """
    Balanced Adjusted Rand Index that accounts for class imbalance.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels
    method : str, default='macro'
        Balancing method:
        - 'macro': Average ARI per class (equal weight to each class)
        - 'weighted': Weight ARI by class size
        - 'sqrt_weighted': Weight by square root of class size (compromise)
        - 'harmonic': Harmonic mean weighting (emphasizes smaller classes)

    Returns:
    --------
    balanced_ari : float
        Balanced Adjusted Rand Index
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    classes = np.unique(labels_true)
    n_classes = len(classes)

    if n_classes == 1:
        return 1.0  # Only one class, perfect clustering

    # Calculate per-class ARI scores
    class_aris = []
    class_sizes = []

    for c in classes:
        # Get indices for this class
        mask = labels_true == c
        class_size = np.sum(mask)
        class_sizes.append(class_size)

        if class_size < 2:
            # Skip classes with less than 2 samples
            continue

        # For this class, calculate how well its samples cluster together
        # We treat this as a binary problem: this class vs others
        true_binary = mask.astype(int)
        pred_for_class = labels_pred[mask]

        # Find the most common predicted cluster for this class
        unique_preds, counts = np.unique(pred_for_class, return_counts=True)

        if len(unique_preds) == 0:
            class_aris.append(0.0)
            continue

        # Calculate purity-based score for this class
        # Higher score if class members are mostly in the same predicted cluster
        max_cluster_size = np.max(counts)
        purity = max_cluster_size / class_size

        # Convert purity to ARI-like scale [-1, 1]
        # Expected purity under random assignment
        n_pred_clusters = len(np.unique(labels_pred))
        expected_purity = 1.0 / n_pred_clusters

        if expected_purity < 1.0:
            class_ari = (purity - expected_purity) / (1.0 - expected_purity)
        else:
            class_ari = 1.0 if purity == 1.0 else 0.0

        class_aris.append(class_ari)

    class_aris = np.array(class_aris)
    class_sizes = np.array(class_sizes[:len(class_aris)])

    # Apply balancing method
    if method == 'macro':
        # Equal weight to each class
        balanced_ari = np.mean(class_aris)

    elif method == 'weighted':
        # Weight by class size (less balanced, but considers size)
        weights = class_sizes / np.sum(class_sizes)
        balanced_ari = np.sum(class_aris * weights)

    elif method == 'sqrt_weighted':
        # Weight by square root of class size (compromise)
        sqrt_sizes = np.sqrt(class_sizes)
        weights = sqrt_sizes / np.sum(sqrt_sizes)
        balanced_ari = np.sum(class_aris * weights)

    elif method == 'harmonic':
        # Harmonic mean weighting (emphasizes smaller classes)
        inv_sizes = 1.0 / class_sizes
        weights = inv_sizes / np.sum(inv_sizes)
        balanced_ari = np.sum(class_aris * weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    return balanced_ari


def class_balanced_ari(labels_true, labels_pred):
    """
    Alternative balanced ARI using per-class F1-like approach.
    Computes ARI for each class treated as binary classification problem,
    then averages (macro-average).

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels

    Returns:
    --------
    balanced_ari : float
        Class-balanced Adjusted Rand Index
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    classes = np.unique(labels_true)
    class_scores = []

    for c in classes:
        # Binary problem: this class vs rest
        y_true_binary = (labels_true == c).astype(int)

        # For predictions, find best matching cluster for this class
        mask = labels_true == c
        if np.sum(mask) == 0:
            continue

        pred_for_class = labels_pred[mask]
        unique_preds, counts = np.unique(pred_for_class, return_counts=True)
        best_cluster = unique_preds[np.argmax(counts)]

        y_pred_binary = (labels_pred == best_cluster).astype(int)

        # Calculate ARI for this binary classification
        ari = adjusted_rand_index_numpy(y_true_binary, y_pred_binary)
        class_scores.append(ari)

    return np.mean(class_scores)


if __name__ == "__main__":
    data, y_true = create_unbalance()

    y_pred = np.copy(y_true)
    y_pred[y_pred > 3] = 3
    from sklearn.metrics import adjusted_rand_score

    print("\nComparison with sklearn:")
    print(f"Sklearn implementation: {adjusted_rand_index(y_true, y_pred):.6f}")
    print(f"Custom implementation: {adjusted_rand_index_numpy(y_true, y_pred):.6f}")
    print(f"Sklearn implementation: {adjusted_rand_score(y_true, y_pred):.6f}")
    print()
    print(f"Custom implementation: {balanced_adjusted_rand_index(y_true, y_pred):.6f}")
    print(f"Custom implementation: {class_balanced_ari(y_true, y_pred):.6f}")
