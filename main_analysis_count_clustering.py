import os
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math

from constants import scale, FOLDER_RESULTS_CORRELATION, FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS, LABEL_COLOR_MAP
from constants_maps import METRICS, MAP_LOWER_IS_BETTER
from load_CVIs import choose_index
from utils import reencode, remove_dups, get_label_files


def create_error_scatter_plot(X, error_cases, best_ari_case, metric, dataset_name, algo_name, lower_is_better, output_folder):
    """
    Create a scatter plot showing all erroneous cases plus the best ARI case.

    Parameters:
    -----------
    X : array
        The dataset (2D for plotting, will use first 2 dimensions if higher)
    error_cases : list of dict
        Each dict contains 'labels', 'ari', 'cvi', 'file'
    best_ari_case : dict
        The parameterization with best ARI (will be highlighted)
    metric : str
        Name of the CVI metric
    dataset_name : str
        Name of the dataset
    algo_name : str
        Name of the clustering algorithm
    lower_is_better : bool
        Whether lower CVI values are better
    output_folder : str
        Folder to save the plots
    """
    if len(error_cases) == 0:
        return

    # Include best ARI case in the plot
    all_cases = [best_ari_case] + error_cases
    n_cases = len(all_cases)

    # Determine grid layout based on number of subplots
    if n_cases <= 3:
        n_cols = n_cases
        n_rows = 1
    elif n_cases <= 8:
        n_cols = 4
        n_rows = math.ceil(n_cases / 4)
    elif n_cases <= 15:
        n_cols = 5
        n_rows = math.ceil(n_cases / 5)
    else:
        n_cols = 6
        n_rows = math.ceil(n_cases / 6)

    # Create figure
    fig_width = min(4 * n_cols, 24)
    fig_height = min(3.5 * n_rows, 20)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easier iteration
    if n_cases == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Use first 2 dimensions for plotting
    X_plot = X[:, :2] if X.shape[1] >= 2 else X

    for idx, (ax, case) in enumerate(zip(axes, all_cases)):
        labels = case['labels']
        ari = case['ari']
        cvi = case['cvi']

        # Plot scatter
        unique_labels = np.unique(labels)

        if idx == 0 and cvi[metric] == 0 and len(unique_labels) == 2:
            print(f"cplm {np.unique(labels)}")

        for label_idx, label in enumerate(unique_labels):
            mask = labels == label
            if len(unique_labels) < len(LABEL_COLOR_MAP.keys()):
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[LABEL_COLOR_MAP[label]], label=f'Cluster {label}', alpha=0.6, s=20)

        # Highlight best ARI case
        if idx == 0:
            title_prefix = "★ BEST ARI ★\n"
            ax.set_facecolor('#ffffcc')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            title_prefix = f"Error {idx}\n"

        # Add title with ARI and CVI
        if cvi[metric] is not None:
            ax.set_title(f"{title_prefix} ARI={ari:.3f}, {metric}={cvi[metric] :.3f}", fontsize=9, fontweight='bold' if idx == 0 else 'normal')
        else:
            ax.set_title(f"{title_prefix} ARI={ari:.3f}, {metric}=None", fontsize=9, fontweight='bold' if idx == 0 else 'normal')

        ax.set_xlabel('Feature 1', fontsize=8)
        ax.set_ylabel('Feature 2', fontsize=8)
        ax.tick_params(labelsize=7)

        # Remove legend if too many clusters
        if len(unique_labels) > 10:
            ax.legend().set_visible(False)
        else:
            ax.legend(fontsize=6, loc='best', framealpha=0.7)

    # Hide unused subplots
    for idx in range(n_cases, len(axes)):
        axes[idx].set_visible(False)

    # Main title
    direction = "lower" if lower_is_better else "higher"
    fig.suptitle(f"{dataset_name} - {algo_name} - {metric}\n"
                 f"({n_cases - 1} errors: {direction} is better)",
                 fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{metric}_{dataset_name}_{algo_name}_errors.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"      Saved error plot: {filepath}")


def compute_best_match_analysis_per_dataset(datasets, metrics, labels_folder, create_plots=True, plot_output_folder=None):
    """
    For each dataset and clustering algorithm:
    1. Find parameterization with highest ARI
    2. Check if that parameterization also has best CVI value
    3. Count correct evaluations (binary: 1 if match, 0 otherwise)
    4. Count errors (number of parameterizations with better CVI + failed parameterizations)
    5. Create scatter plots for erroneous cases

    Returns:
    --------
    dict : Dictionary containing results aggregated by dataset and by clustering algorithm
    """
    results_by_dataset = {}
    results_by_algo = {}

    for dataset_name, (X, labels_gt) in datasets:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Preprocess data
        X = MinMaxScaler(scale).fit_transform(X)
        X, labels_gt = remove_dups(X, labels_gt)
        labels_gt_re = reencode(labels_gt)

        # Find all label files for this dataset
        pattern = os.path.join(labels_folder, "labels_*.npy")
        label_files = get_label_files(pattern, dataset_name)

        if len(label_files) == 0:
            print(f"Warning: No label files found for {dataset_name} with pattern {pattern}")
            continue

        print(f"Found {len(label_files)} clustering algorithm results")

        # Group by clustering algorithm
        algo_groups = {}
        for label_file in label_files:
            filename = Path(label_file).stem
            # Extract algorithm name (assumes format: labels_<dataset>_<algo>_<params>)
            parts = filename.replace(f"labels_{dataset_name}_", "").split("_")
            algo_name = parts[-2] if len(parts) > 1 else "unknown"

            if algo_name not in algo_groups:
                algo_groups[algo_name] = []
            algo_groups[algo_name].append(label_file)

        # Storage for this dataset
        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = {metric: {'correct': 0, 'errors': 0} for metric in metrics}

        # Process each clustering algorithm
        for algo_name, algo_files in algo_groups.items():
            print(f"\n  Processing algorithm: {algo_name} ({len(algo_files)} parameterizations)")

            # Storage for all parameterizations of this algorithm
            param_results = []

            for label_file in algo_files:
                try:
                    labels_clustering = np.load(label_file)
                except Exception as e:
                    print(f"    Warning: Failed to load {label_file}: {e}")
                    continue

                # Skip single-cluster results - this is also a failure
                unique_labels = np.unique(labels_clustering)
                if len(unique_labels) == 1 or (-1 in unique_labels and len(unique_labels) <= 2):
                    continue

                labels_clustering_re = reencode(labels_clustering)

                if len(labels_clustering_re) != len(labels_gt_re):
                    continue

                ari_value = adjusted_rand_score(labels_gt_re, labels_clustering_re)

                # Compute all CVIs
                cvi_results = {}
                for metric in metrics:
                    try:
                        cvi_value = choose_index(metric=metric, data=X, labels=labels_clustering)
                        if np.isnan(cvi_value) or np.isinf(cvi_value):
                            cvi_results[metric] = None
                        else:
                            cvi_results[metric] = cvi_value
                    except Exception as e:
                        print(f"    Warning: Failed to compute {metric}: {e}")
                        cvi_results[metric] = None

                param_results.append({
                    'file': label_file,
                    'ari': ari_value,
                    'labels': labels_clustering,
                    'cvi': cvi_results
                })

            if len(param_results) < 2:
                print(f"    Skipping {algo_name}: insufficient valid parameterizations")
                continue

            # Find parameterization with highest ARI
            best_ari_idx = np.argmax([p['ari'] for p in param_results])
            best_ari_param = param_results[best_ari_idx]

            print(f"    Best ARI: {best_ari_param['ari']:.4f}")

            # Initialize results for this algorithm if not exists
            if algo_name not in results_by_algo:
                results_by_algo[algo_name] = {metric: {'correct': 0, 'errors': 0} for metric in metrics}

            # For each CVI, check if best ARI also gives best CVI
            for metric in metrics:
                # Collect valid CVI values
                valid_cvi_values = [(i, p['cvi'][metric]) for i, p in enumerate(param_results)
                                    if p['cvi'][metric] is not None]

                if len(valid_cvi_values) < 2:
                    # Not enough valid values - count failed ones as errors
                    failed_count = len(param_results) - len(valid_cvi_values)
                    results_by_dataset[dataset_name][metric]['errors'] += failed_count
                    results_by_algo[algo_name][metric]['errors'] += failed_count
                    print(f"      {metric}: ERROR (count={failed_count})")
                    continue

                # Determine best CVI index
                lower_is_better = True if metric.lower() in MAP_LOWER_IS_BETTER else False

                if lower_is_better:
                    best_cvi_idx = min(valid_cvi_values, key=lambda x: x[1])[0]
                else:
                    best_cvi_idx = max(valid_cvi_values, key=lambda x: x[1])[0]

                # Check if best ARI matches best CVI
                if best_ari_idx == best_cvi_idx:
                    # CORRECT - binary count of 1
                    results_by_dataset[dataset_name][metric]['correct'] += 1
                    results_by_algo[algo_name][metric]['correct'] += 1
                    print(f"      {metric}: CORRECT")
                else:
                    # NOT CORRECT - count how many parameterizations have better CVI
                    best_ari_cvi = best_ari_param['cvi'][metric]

                    error_cases = []

                    if best_ari_cvi is None:
                        # Best ARI param failed for this CVI - all others are errors
                        error_count = len(param_results)
                        error_cases = [p for i, p in enumerate(param_results) if i != best_ari_idx]
                    else:
                        # Count parameterizations with better CVI than best ARI one
                        error_count = 0
                        for idx, cvi_val in valid_cvi_values:
                            if idx == best_ari_idx:
                                continue

                            if lower_is_better:
                                if cvi_val < best_ari_cvi:
                                    error_count += 1
                                    error_cases.append(param_results[idx])
                            else:
                                if cvi_val > best_ari_cvi:
                                    error_count += 1
                                    error_cases.append(param_results[idx])

                        # Also count failed parameterizations as errors
                        failed_count = len(param_results) - len(valid_cvi_values)
                        error_count += failed_count

                    results_by_dataset[dataset_name][metric]['errors'] += error_count
                    results_by_algo[algo_name][metric]['errors'] += error_count
                    print(f"      {metric}: ERROR (count={error_count})")

                    # Create visualization for errors
                    if create_plots and plot_output_folder and len(error_cases) > 0:
                        create_error_scatter_plot(
                            X=X,
                            error_cases=error_cases,
                            best_ari_case=best_ari_param,
                            metric=metric,
                            dataset_name=dataset_name,
                            algo_name=algo_name,
                            lower_is_better=lower_is_better,
                            output_folder=plot_output_folder
                        )

    return results_by_dataset, results_by_algo


def save_results(results_by_dataset, results_by_algo, file_prefix):
    dataset_rows = []
    for dataset, metrics_dict in results_by_dataset.items():
        for metric, counts in metrics_dict.items():
            dataset_rows.append({
                'dataset': dataset,
                'metric': metric,
                'correct': counts['correct'],
                'errors': counts['errors']
            })

    df_by_dataset = pd.DataFrame(dataset_rows)
    df_by_dataset_pivot = df_by_dataset.pivot(index='metric', columns='dataset',
                                              values=['correct', 'errors'])

    output_path_dataset = FOLDER_RESULTS_CORRELATION + f"{file_prefix}_best_match_by_dataset.csv"
    df_by_dataset_pivot.to_csv(output_path_dataset)
    print(f"\n>>> Results by dataset saved to: {output_path_dataset}")

    # Create DataFrame aggregated by clustering algorithm
    algo_rows = []
    for algo, metrics_dict in results_by_algo.items():
        for metric, counts in metrics_dict.items():
            algo_rows.append({
                'algorithm': algo,
                'metric': metric,
                'correct': counts['correct'],
                'errors': counts['errors']
            })

    df_by_algo = pd.DataFrame(algo_rows)
    df_by_algo_pivot = df_by_algo.pivot(index='metric', columns='algorithm', values=['correct', 'errors'])

    output_path_algo = FOLDER_RESULTS_CORRELATION + f"{file_prefix}_best_match_by_algorithm.csv"
    df_by_algo_pivot.to_csv(output_path_algo)
    print(f">>> Results by algorithm saved to: {output_path_algo}")


def main_real_data():
    from load_datasets import create_real_datasets_uci

    datasets = create_real_datasets_uci()
    prefix = "realdata"
    plot_folder = FOLDER_RESULTS_CORRELATION + f"{prefix}_error_plots/"

    results_by_dataset, results_by_algo = compute_best_match_analysis_per_dataset(
        datasets=datasets,
        metrics=METRICS if "CDbw" not in METRICS else [m for m in METRICS if m != "CDbw"],
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
        create_plots=True,
        plot_output_folder=plot_folder
    )

    save_results(results_by_dataset, results_by_algo, prefix)


def main_synth_data():
    from load_datasets import create_synthetic_datasets

    datasets = create_synthetic_datasets()
    prefix = "synthdata"
    plot_folder = FOLDER_RESULTS_CORRELATION + f"{prefix}_error_plots/"

    results_by_dataset, results_by_algo = compute_best_match_analysis_per_dataset(
        datasets=datasets,
        metrics=METRICS,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
        create_plots=True,
        plot_output_folder=plot_folder
    )

    save_results(results_by_dataset, results_by_algo, prefix)


if __name__ == "__main__":
    main_synth_data()
    # main_real_data()
