import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

from constants import scale, FOLDER_RESULTS_CORRELATION, FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS
from constants_maps import METRICS, MAP_LOWER_IS_BETTER
from cvis_ours.external_CVIs import balanced_external
from load_CVIs import choose_index
from utils import reencode, remove_dups, get_label_files



def compute_ari_cvi_correlations_per_dataset(datasets, metrics, labels_folder):
    """
    Compute Pearson correlation between ARI/BARI and internal CVIs for each dataset.
    Also compute versions excluding noise points (_nn versions).

    For each dataset:
    - Find all label files matching "labels_<dataset_name>_*.npy"
    - For each CVI metric:
        - Compute CVI and ARI values across all clustering algorithms
        - Compute correlation between CVI and ARI

    Parameters:
    -----------
    datasets : list of tuples
        List of (dataset_name, (data, ground_truth_labels))
    metrics : list
        List of CVI metric names
    labels_folder : str
        Path to directory containing clustering labels

    Returns:
    --------
    dict : Dictionary containing 4 DataFrames (ari, ari_nn, bari, bari_nn)
    """
    results_ari = {}
    results_ari_nn = {}
    results_bari = {}
    results_bari_nn = {}

    for dataset_name, (X, labels_gt) in datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Preprocess data
        X = MinMaxScaler(scale).fit_transform(X)
        X, labels_gt = remove_dups(X, labels_gt)
        labels_gt_re = reencode(labels_gt)

        # Find all label files for this dataset
        pattern = f"{labels_folder}/labels_{dataset_name}_*.npy"
        label_files = get_label_files(pattern, dataset_name)

        if len(label_files) == 0:
            print(f"Warning: No label files found for {dataset_name} with pattern {pattern}")
            continue

        print(f"Found {len(label_files)} clustering algorithm results")

        # Storage for this dataset
        ari_values = []
        ari_nn_values = []
        bari_values = []
        bari_nn_values = []

        cvi_values = {metric: [] for metric in metrics}
        cvi_nn_values = {metric: [] for metric in metrics}

        # Process each clustering algorithm result
        for label_file in label_files:
            clusterer_name = Path(label_file).stem.replace(f"labels_{dataset_name}_", "")

            try:
                labels_clustering = np.load(label_file)
            except Exception as e:
                print(f"Warning: Failed to load {label_file}: {e}")
                continue

            if len(np.unique(labels_clustering)) == 1:
                continue

            labels_clustering_re = reencode(labels_clustering)

            # Prepare no-noise versions
            X_nn = X[labels_clustering != -1]
            labels_gt_nn = labels_gt[labels_clustering != -1]
            labels_clustering_nn = labels_clustering[labels_clustering != -1]

            if len(np.unique(labels_clustering_nn)) == 1:
                continue

            # Compute ARI
            ari_values.append(adjusted_rand_score(labels_gt_re, labels_clustering_re))
            ari_nn_values.append(adjusted_rand_score(labels_gt_nn, labels_clustering_nn))

            # Compute BARI
            bari_values.append(balanced_external(adjusted_rand_score, labels_gt_re, labels_clustering_re, method='macro'))
            bari_nn_values.append(balanced_external(adjusted_rand_score, labels_gt_nn, labels_clustering_nn, method='macro'))

            # Compute each CVI
            for metric in metrics:
                try:
                    cvi_values[metric].append(choose_index(metric=metric, data=X, labels=labels_clustering))
                    cvi_nn_values[metric].append(choose_index(metric=metric, data=X_nn, labels=labels_clustering_nn))
                except Exception as e:
                    print(f"Warning: Failed to compute {metric} for {clusterer_name}: {e}")
                    cvi_values[metric].append(np.nan)
                    cvi_nn_values[metric].append(np.nan)

            # print(f"  - {clusterer_name}: ARI={ari_values[-1]:.3f}, noise={np.count_nonzero(labels_clustering == -1)}/{len(labels_clustering)}")

        # Compute correlations for this dataset
        dataset_correlations_ari = {}
        dataset_correlations_ari_nn = {}
        dataset_correlations_bari = {}
        dataset_correlations_bari_nn = {}
        for metric in metrics:
            dataset_correlations_ari[metric] = compute_single_correlation(metric, ari_values, cvi_values[metric], dataset_name)
            dataset_correlations_ari_nn[metric] = compute_single_correlation(metric, ari_nn_values, cvi_nn_values[metric], dataset_name)
            dataset_correlations_bari[metric] = compute_single_correlation(metric, bari_values, cvi_values[metric], dataset_name)
            dataset_correlations_bari_nn[metric] = compute_single_correlation(metric, bari_nn_values, cvi_nn_values[metric], dataset_name)

        results_ari[dataset_name] = dataset_correlations_ari
        results_ari_nn[dataset_name] = dataset_correlations_ari_nn
        results_bari[dataset_name] = dataset_correlations_bari
        results_bari_nn[dataset_name] = dataset_correlations_bari_nn

    return {
        'ari': pd.DataFrame(results_ari),
        'ari_nn': pd.DataFrame(results_ari_nn),
        'bari': pd.DataFrame(results_bari),
        'bari_nn': pd.DataFrame(results_bari_nn)
    }


def compute_single_correlation(metric, external_vals, cvi_vals, dataset_name):
    """Helper function to compute a single correlation"""
    external_arr = np.array(external_vals)
    cvi_arr = np.array(cvi_vals)

    # Filter out NaN/inf values
    valid_indices = ~np.isnan(cvi_arr) & np.isfinite(cvi_arr) & ~np.isnan(external_arr) & np.isfinite(external_arr)

    if np.sum(valid_indices) < 2:
        print(f"Warning: Insufficient valid data for metric {metric} on {dataset_name}")
        return np.nan

    if np.sum(~valid_indices) > 0:
        print(f"Warning: metric {metric} contains {np.sum(~valid_indices)} NaN/inf values")

    external_valid = np.array(external_vals)[valid_indices]
    cvi_valid = np.array(cvi_vals)[valid_indices]

    # Compute Pearson correlation
    # corr, p_value = pearsonr(external_valid, cvi_valid)
    corr, p_value = spearmanr(external_valid, cvi_valid)
    print(f"  {metric}: correlation={corr:.3f}, p-value={p_value:.4f}")

    if metric.lower() in MAP_LOWER_IS_BETTER:
        return -corr
    else:
        return corr


def save_correlation_matrix(df, file_name="correlations_cvi_to_ari_per_dataset"):
    """Save correlation matrix to CSV"""
    output_path = FOLDER_RESULTS_CORRELATION + f"{file_name}.csv"
    df.to_csv(output_path)
    print(f"\n>>> Results saved to: {output_path}")
    print(f">>> Shape: {df.shape[0]} CVIs x {df.shape[1]} datasets")


def main_real_data():
    from load_datasets import create_real_datasets

    datasets = create_real_datasets()

    prefix = "realdata"
    # Compute correlations per dataset
    correlation_matrices = compute_ari_cvi_correlations_per_dataset(
        datasets=datasets,
        metrics=METRICS if "CDbw" not in METRICS else [m for m in METRICS if m != "CDbw"],
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
    )

    # Save results
    save_correlation_matrix(correlation_matrices['ari'],        file_name=f"{prefix}_correlations_cvi_to_ari")
    save_correlation_matrix(correlation_matrices['ari_nn'],     file_name=f"{prefix}_correlations_cvi_to_ari_nn")
    save_correlation_matrix(correlation_matrices['bari'],       file_name=f"{prefix}_correlations_cvi_to_bari")
    save_correlation_matrix(correlation_matrices['bari_nn'],    file_name=f"{prefix}_correlations_cvi_to_bari_nn")


def main_synth_data():
    from load_datasets import create_synthetic_datasets

    datasets = create_synthetic_datasets()

    prefix = "synthdata"
    # Compute correlations per dataset
    correlation_matrices = compute_ari_cvi_correlations_per_dataset(
        datasets=datasets,
        metrics=METRICS,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
    )

    # Save results
    save_correlation_matrix(correlation_matrices['ari'],        file_name=f"{prefix}_correlations_cvi_to_ari")
    save_correlation_matrix(correlation_matrices['ari_nn'],     file_name=f"{prefix}_correlations_cvi_to_ari_nn")
    save_correlation_matrix(correlation_matrices['bari'],       file_name=f"{prefix}_correlations_cvi_to_bari")
    save_correlation_matrix(correlation_matrices['bari_nn'],    file_name=f"{prefix}_correlations_cvi_to_bari_nn")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected."
    )

    main_real_data()
    main_synth_data()

