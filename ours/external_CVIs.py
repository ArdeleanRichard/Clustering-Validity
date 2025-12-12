import numpy as np
from scipy.special import comb


def adjusted_rand_index_python(labels_true, labels_pred):
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


def adjusted_rand_index_numpy(labels_true, labels_pred):
    """
    Optimized NumPy implementation of Adjusted Rand Index.
    Pure NumPy implementation (faster for large datasets).

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


def method_balance(method="macro", class_scores=None, class_sizes=None):
    # Apply balancing method
    if method == 'macro':
        # Equal weight to each class
        balanced_ari = np.mean(class_scores)

    elif method == 'weighted':
        # Weight by class size (less balanced, but considers size)
        weights = class_sizes / np.sum(class_sizes)
        balanced_ari = np.sum(class_scores * weights)

    elif method == 'sqrt_weighted':
        # Weight by square root of class size (compromise)
        sqrt_sizes = np.sqrt(class_sizes)
        weights = sqrt_sizes / np.sum(sqrt_sizes)
        balanced_ari = np.sum(class_scores * weights)

    elif method == 'harmonic':
        # Harmonic mean weighting (emphasizes smaller classes)
        inv_sizes = 1.0 / class_sizes
        weights = inv_sizes / np.sum(inv_sizes)
        balanced_ari = np.sum(class_scores * weights)

    else:
        raise ValueError(f"Unknown method: {method}")

    return balanced_ari


def balanced_external(cvi, labels_true, labels_pred, method='macro'):
    """
    Alternative balanced CVI using per-class F1-like approach.
    Computes CVI for each class treated as binary classification problem, then averages.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels

    Returns:
    --------
    balanced_score : float
        Class-balanced CVI
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    classes = np.unique(labels_true)
    n_classes = len(classes)

    if n_classes == 1:
        return 1.0  # Only one class, perfect clustering

    # Calculate per-class ARI scores
    class_scores = []
    class_sizes = []

    for c in classes:
        # Binary problem: this class vs rest
        mask = labels_true == c
        class_size = np.sum(mask)
        class_sizes.append(class_size)

        y_true_binary = mask.astype(int)

        # For predictions, find best matching cluster for this class

        if np.sum(mask) == 0:
            continue

        pred_for_class = labels_pred[mask]
        unique_preds, counts = np.unique(pred_for_class, return_counts=True)
        best_cluster = unique_preds[np.argmax(counts)]

        y_pred_binary = (labels_pred == best_cluster).astype(int)

        # Calculate ARI for this binary classification
        cvi_score = cvi(y_true_binary, y_pred_binary)
        class_scores.append(cvi_score)

    class_scores = np.array(class_scores)
    class_sizes = np.array(class_sizes[:len(class_scores)])

    balanced_score = method_balance(method=method, class_scores=class_scores, class_sizes=class_sizes)

    return balanced_score

#
# def balanced_sample_weighted(cvi, labels_true, labels_pred, weight_method='inverse_freq'):
#     """
#     Sample-weighted balanced metric.
#     Assigns weights to samples inversely proportional to class size.
#
#     Parameters:
#     -----------
#     cvi : callable
#         External clustering metric (must support sample weights or use weighted pairs)
#     labels_true : array-like
#         Ground truth labels
#     labels_pred : array-like
#         Predicted labels
#     weight_method : str
#         'inverse_freq', 'sqrt_inverse_freq', or 'log_inverse_freq'
#
#     Returns:
#     --------
#     weighted_score : float
#     """
#     labels_true = np.asarray(labels_true)
#     labels_pred = np.asarray(labels_pred)
#
#     classes, class_counts = np.unique(labels_true, return_counts=True)
#     n_samples = len(labels_true)
#
#     # Compute class weights
#     if weight_method == 'inverse_freq':
#         class_weights = n_samples / (len(classes) * class_counts)
#     elif weight_method == 'sqrt_inverse_freq':
#         class_weights = np.sqrt(n_samples / (len(classes) * class_counts))
#     elif weight_method == 'log_inverse_freq':
#         class_weights = np.log(n_samples / class_counts + 1)
#     else:
#         raise ValueError(f"Unknown weight_method: {weight_method}")
#
#     # Map weights to samples
#     weight_map = dict(zip(classes, class_weights))
#     sample_weights = np.array([weight_map[c] for c in labels_true])
#
#     # For metrics that don't support weights natively,
#     # we use weighted bootstrap resampling
#     n_bootstrap = 100
#     scores = []
#
#     for _ in range(n_bootstrap):
#         # Resample with probability proportional to weights
#         probs = sample_weights / sample_weights.sum()
#         indices = np.random.choice(n_samples, size=n_samples, replace=True, p=probs)
#         scores.append(cvi(labels_true[indices], labels_pred[indices]))
#
#     return np.mean(scores)
#
# def balanced_stratified_subsample(cvi, labels_true, labels_pred, n_samples_per_class='min', n_iterations: int = 10):
#     """
#     Stratified subsampling balanced metric.
#     Samples equal numbers from each class and averages metric over iterations.
#
#     Parameters:
#     -----------
#     cvi : callable
#         External clustering metric
#     labels_true : array-like
#         Ground truth labels
#     labels_pred : array-like
#         Predicted labels
#     n_samples_per_class : int or 'min'
#         Number of samples per class, or 'min' for minimum class size
#     n_iterations : int
#         Number of subsampling iterations
#
#     Returns:
#     --------
#     stratified_score : float
#     """
#     labels_true = np.asarray(labels_true)
#     labels_pred = np.asarray(labels_pred)
#
#     classes = np.unique(labels_true)
#     class_sizes = [np.sum(labels_true == c) for c in classes]
#
#     if n_samples_per_class == 'min':
#         n_samples = min(class_sizes)
#     else:
#         n_samples = n_samples_per_class
#
#     scores = []
#     for _ in range(n_iterations):
#         sampled_indices = []
#         for c in classes:
#             class_indices = np.where(labels_true == c)[0]
#             n_to_sample = min(n_samples, len(class_indices))
#             sampled = np.random.choice(class_indices, size=n_to_sample, replace=False)
#             sampled_indices.extend(sampled)
#
#         sampled_indices = np.array(sampled_indices)
#         scores.append(cvi(labels_true[sampled_indices], labels_pred[sampled_indices]))
#
#     return np.mean(scores)
#
# def balanced_pairwise_classes(cvi, labels_true, labels_pred, mode='all_pairs'):
#     """
#     Pairwise class comparison balanced metric.
#
#     Parameters:
#     -----------
#     cvi : callable
#         External clustering metric
#     labels_true : array-like
#         Ground truth labels
#     labels_pred : array-like
#         Predicted labels
#     mode : str
#         'all_pairs': Average over all class pairs
#         'within_between': Separate within-class and between-class scores
#
#     Returns:
#     --------
#     pairwise_score : float
#     """
#     labels_true = np.asarray(labels_true)
#     labels_pred = np.asarray(labels_pred)
#
#     classes = np.unique(labels_true)
#
#     if mode == 'all_pairs':
#         pair_scores = []
#         for i, c1 in enumerate(classes):
#             for c2 in classes[i:]:
#                 mask = (labels_true == c1) | (labels_true == c2)
#                 if np.sum(mask) < 2:
#                     continue
#                 score = cvi(labels_true[mask], labels_pred[mask])
#                 pair_scores.append(score)
#
#         return np.mean(pair_scores) if pair_scores else 0.0
#
#     elif mode == 'within_between':
#         within_scores = []
#         between_scores = []
#
#         # Within-class scores
#         for c in classes:
#             mask = labels_true == c
#             if np.sum(mask) < 2:
#                 continue
#             within_scores.append(cvi(labels_true[mask], labels_pred[mask]))
#
#         # Between-class scores
#         for i, c1 in enumerate(classes):
#             for c2 in classes[i + 1:]:
#                 mask = (labels_true == c1) | (labels_true == c2)
#                 if np.sum(mask) < 2:
#                     continue
#                 between_scores.append(cvi(labels_true[mask], labels_pred[mask]))
#
#         within_avg = np.mean(within_scores) if within_scores else 0.0
#         between_avg = np.mean(between_scores) if between_scores else 0.0
#
#         return (within_avg + between_avg) / 2
#
#     else:
#         raise ValueError(f"Unknown mode: {mode}")
#
# def balanced_minmax(cvi, labels_true, labels_pred, alpha=0.5):
#     """
#     Min-max balanced metric.
#     Balances between average and worst-case class performance.
#
#     Parameters:
#     -----------
#     cvi : callable
#         External clustering metric
#     labels_true : array-like
#         Ground truth labels
#     labels_pred : array-like
#         Predicted labels
#     alpha : float, default=0.5
#         Balance parameter:
#         - 0.0: Only worst class matters (min)
#         - 1.0: Only average matters (mean)
#         - 0.5: Equal balance
#
#     Returns:
#     --------
#     minmax_score : float
#     """
#     labels_true = np.asarray(labels_true)
#     labels_pred = np.asarray(labels_pred)
#
#     classes = np.unique(labels_true)
#     class_scores = []
#
#     for c in classes:
#         mask = labels_true == c
#         if np.sum(mask) < 2:
#             continue
#
#         y_true_binary = mask.astype(int)
#         pred_for_class = labels_pred[mask]
#         unique_preds, counts = np.unique(pred_for_class, return_counts=True)
#         best_cluster = unique_preds[np.argmax(counts)]
#         y_pred_binary = (labels_pred == best_cluster).astype(int)
#
#         score = cvi(y_true_binary, y_pred_binary)
#         class_scores.append(score)
#
#     if not class_scores:
#         return 0.0
#
#     class_scores = np.array(class_scores)
#     min_score = np.min(class_scores)
#     mean_score = np.mean(class_scores)
#
#     return alpha * mean_score + (1 - alpha) * min_score
#
#
#     # Normalize weights
#     class_weights = class_weights / np.sum(class_weights)
#
#     return np.sum(class_scores * class_weights)