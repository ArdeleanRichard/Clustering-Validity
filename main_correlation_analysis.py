import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

from constants import scale, FOLDER_RESULTS_CLUSTERING_LABELS
from constants_maps import METRICS
from load_CVIs import choose_index
from utils import reencode, remove_dups


def compute_ari_cvi_correlations(datasets, metrics, labels_path=FOLDER_RESULTS_CLUSTERING_LABELS):
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

    # Storage for ARI and CVI values across all datasets
    ari_values = []
    cvi_values = {metric: [] for metric in metrics}

    # Iterate through each dataset
    for dataset_name, (X, gt) in datasets:
        X = MinMaxScaler(scale).fit_transform(X)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)

        # Load clustering labels
        labels_file = f"{labels_path}labels_{dataset_name}_DBSCAN.npy"

        try:
            clustering_labels = np.load(labels_file)
            clustering_labels = reencode(clustering_labels) # SOME METRIC GIVE OTHERWISE NaN due to -1 noise
        except FileNotFoundError:
            print(f"Warning: Labels file not found for {dataset_name}, skipping...")
            continue

        # Compute ARI
        ari = adjusted_rand_score(gt, clustering_labels)
        ari_values.append(ari)

        # Compute each CVI
        for metric in metrics:
            try:
                # unique_labels = np.unique(clustering_labels, return_counts=True)
                cvi_value = choose_index(metric=metric, data=X, labels=clustering_labels)
                cvi_values[metric].append(cvi_value)
                # print(cvi_value, list(zip(unique_labels[0], unique_labels[1])))
            except Exception as e:
                print(f"Warning: Failed to compute {metric} for {dataset_name}: {e}")
                cvi_values[metric].append(np.nan)

        print(f"Processed {dataset_name}: ARI = {ari:.4f}")

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



    return correlations, ari_values, cvi_values


# Usage example:
if __name__ == "__main__":
    from load_datasets import create_compound, create_aggregation, create_jain, create_unbalance, create_spiral, create_pathbased, \
        create_data1, create_data2, create_data3, create_data4, create_data5, create_data6, create_data7, \
        create_parabolic, create_ring, create_zigzag, create_trajectories, create_x, create_set_s, create_set_a, create_d31
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected."
    )

    n_samples = 1000
    datasets = [
        ("data1", create_data1(n_samples)),
        ("data2", create_data2(n_samples)),
        ("data3", create_data3(n_samples)),
        ("data4", create_data4(n_samples)),
        ("data5", create_data5(n_samples)),
        ("data6", create_data6(n_samples)),
        ("data7", create_data7(n_samples)),
        ("aggregation", create_aggregation()),
        ("compound", create_compound()),
        ("d31", create_d31()),
        ("jain", create_jain()),
        ("pathbased", create_pathbased()),
        ("spiral", create_spiral()),
        ("unbalance", create_unbalance()),
    ]
    datasets.extend([("parabolic", create_parabolic())])
    datasets.extend([(f"ring{t}", create_ring(t)) for t in ["", "_noisy", "_outliers"]])
    datasets.extend([(f"zigzag{t}", create_zigzag(t)) for t in ["", "_noisy", "_outliers"]])
    datasets.extend([("trajectories", create_trajectories())])
    datasets.extend([(f"x{i}", create_x(i)) for i in [1, 2, 3]])
    datasets.extend(create_set_s())
    datasets.extend(create_set_a())


    # Compute correlations
    correlations, ari_vals, cvi_vals = compute_ari_cvi_correlations(datasets, METRICS)

    # Create DataFrame and save to CSV
    import pandas as pd

    results_df = pd.DataFrame([
        {'Metric': metric, 'Correlation': corr, 'P-value': p_val}
        for metric, (corr, p_val) in correlations.items()
    ])

    # Sort by absolute correlation
    # results_df['Abs_Correlation'] = results_df['Correlation'].abs()
    # results_df = results_df.sort_values('Abs_Correlation', ascending=False)
    # results_df = results_df.drop('Abs_Correlation', axis=1)

    # Save to CSV
    output_path = "./results/correlations_cvi_to_ari.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print results
    print("\n" + "=" * 60)
    print("Pearson Correlation between ARI and Internal CVIs")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
