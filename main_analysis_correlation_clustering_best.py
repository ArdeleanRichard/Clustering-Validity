import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

from constants import scale, FOLDER_RESULTS_CLUSTERING_LABELS, FOLDER_RESULTS_CORRELATION
from constants_maps import CVIs
from cvis_ours.external_CVIs import balanced_external
from load_CVIs import choose_CVI
from utils import reencode, remove_dups


def compute_correlations(metrics, ari_values, cvi_values):
    # Compute Pearson correlations
    correlations = {}

    for metric in metrics:
        # Filter out NaN/inf values
        valid_indices = ~np.isnan(cvi_values[metric]) & np.isfinite(cvi_values[metric])

        if np.sum(valid_indices) < 2:
            print(f"Warning: Insufficient valid data for metric {metric}")
            correlations[metric] = (np.nan, np.nan)
            continue

        if np.sum(~valid_indices) > 0:
            print(f"Warning: metric {metric} contains NaN/inf values")

        ari_valid = np.array(ari_values)[valid_indices]
        cvi_valid = np.array(cvi_values[metric])[valid_indices]

        # Compute Pearson correlation
        corr, p_value = pearsonr(ari_valid, cvi_valid)
        correlations[metric] = (corr, p_value)

    return correlations

def compute_ari_cvi_correlations(datasets, metrics, clusterer, labels_path=FOLDER_RESULTS_CLUSTERING_LABELS):
    """
    Compute Pearson correlation between ARI and internal CVIs across datasets.

    Parameters:
    -----------
    datasets : list of tuples
        List of (dataset_name, (data, ground_truth_labels))
    metrics : list
        List of CVI metric names
    labels_path : str
        Path to directory containing clustering labels

    Returns:
    --------
    dict : Dictionary with metric names as keys and (correlation, p_value) as values
    """

    ari_values = []
    ari_nn_values = []
    bari_values = []
    bari_nn_values = []

    cvi_values = {metric: [] for metric in metrics}
    cvi_nn_values = {metric: [] for metric in metrics}

    # Iterate through each dataset
    for data_id, (dataset_name, (X, labels_gt)) in enumerate(datasets):
        print(f"Data - {dataset_name}: {data_id+1}/{len(datasets)}")
        X = MinMaxScaler(scale).fit_transform(X)
        X, labels_gt = remove_dups(X, labels_gt)

        # Load clustering labels
        labels_file = f"{labels_path}labels_{dataset_name}_{clusterer}.npy"

        try:
            labels_clustering = np.load(labels_file)
        except FileNotFoundError:
            print(f"Warning: Labels file not found for {dataset_name}, skipping...")
            continue

        labels_clustering_re = reencode(labels_clustering) # SOME METRIC GIVE OTHERWISE NaN due to -1 noise
        # print(np.unique(labels_clustering, return_counts=True))
        # print(np.unique(labels_clustering_re, return_counts=True))
        labels_gt_re = reencode(labels_gt)

        X_nn = X[labels_clustering!=-1]
        labels_gt_nn = labels_gt[labels_clustering!=-1]
        labels_clustering_nn = labels_clustering[labels_clustering!=-1]

        # Compute ARI
        ari_values.append(adjusted_rand_score(labels_gt_re, labels_clustering_re))
        ari_nn_values.append(adjusted_rand_score(labels_gt_nn, labels_clustering_nn))

        bari_values.append(balanced_external(adjusted_rand_score, labels_gt_re, labels_clustering_re, method='macro'))
        bari_nn_values.append(balanced_external(adjusted_rand_score, labels_gt_nn, labels_clustering_nn, method='macro'))

        # Compute each CVI
        for metric_id, metric in enumerate(metrics):
            print(f"\tMetric - {metric}: {metric_id+1}/{len(metrics)}")

            try:
                cvi_values[metric].append(choose_CVI(cvi=metric, data=X, labels=labels_clustering))
                cvi_nn_values[metric].append(choose_CVI(cvi=metric, data=X_nn, labels=labels_clustering_nn))

            except Exception as e:
                print(f"Warning: Failed to compute {metric} for {dataset_name}: {e}")
                cvi_values[metric].append(np.nan)
                cvi_nn_values[metric].append(np.nan)

        print(f"Processed {dataset_name}: noise {np.count_nonzero(labels_clustering==-1)}/{len(labels_clustering)}")

    correlations_ari = compute_correlations(metrics, ari_values, cvi_values)
    correlations_ari_nn = compute_correlations(metrics, ari_nn_values, cvi_nn_values)
    correlations_bari = compute_correlations(metrics, bari_values, cvi_values)
    correlations_bari_nn = compute_correlations(metrics, bari_nn_values, cvi_nn_values)

    save_csv(correlations_ari,      file_name=f"correlations_cvi_to_ari_{clusterer}")
    save_csv(correlations_ari_nn,   file_name=f"correlations_cvi_to_ari_nn_{clusterer}")
    save_csv(correlations_bari,     file_name=f"correlations_cvi_to_bari_{clusterer}")
    save_csv(correlations_bari_nn,  file_name=f"correlations_cvi_to_bari_nn_{clusterer}")


def save_csv(correlations, file_name="correlations_cvi_to_ari"):
    import pandas as pd

    results_df = pd.DataFrame([
        {'Metric': metric, 'Correlation': corr, 'P-value': p_val}
        for metric, (corr, p_val) in correlations.items()
    ])

    # Save to CSV
    output_path = FOLDER_RESULTS_CORRELATION + f"{file_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main_synth_data():
    from load_datasets import create_synthetic_datasets

    datasets = create_synthetic_datasets()

    # Compute correlations
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="KMeans")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="MeanShift")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="DBSCAN")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="HDBSCAN")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="AgglomerativeClustering")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="SpectralClustering")

def main_real_data():
    from load_datasets import create_real_datasets_uci

    datasets = create_real_datasets_uci()

    metrics = CVIs.copy()
    metrics.remove("CDbw") # cannot construct hull # Failed to compute CDbw: QH6214 qhull input error: not enough points to construct initial simplex

    # Compute correlations
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="KMeans")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="MeanShift")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="DBSCAN")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="HDBSCAN")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="AgglomerativeClustering")
    compute_ari_cvi_correlations(datasets, CVIs, clusterer="SpectralClustering")


def main_real_data_new():
    from load_datasets import create_real_datasets_image

    datasets = create_real_datasets_image()

    metrics = CVIs.copy()
    metrics.remove("rCIP") # Failed to compute rCIP: (..., 'Result too large')
    metrics.remove("CDbw") # Failed to compute CDbw: QH6214 qhull input error: not enough points to construct initial simplex

    # Compute correlations
    compute_ari_cvi_correlations(datasets, metrics, clusterer="KMeans")
    compute_ari_cvi_correlations(datasets, metrics, clusterer="MeanShift")
    compute_ari_cvi_correlations(datasets, metrics, clusterer="DBSCAN")
    compute_ari_cvi_correlations(datasets, metrics, clusterer="HDBSCAN")
    compute_ari_cvi_correlations(datasets, metrics, clusterer="AgglomerativeClustering")
    compute_ari_cvi_correlations(datasets, metrics, clusterer="SpectralClustering")


if __name__ == "__main__":
    # main_synth_data()
    # main_real_data()
    main_real_data_new()


