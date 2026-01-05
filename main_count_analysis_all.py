import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
import glob

from constants import scale, FOLDER_RESULTS_CORRELATION, FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS
from constants_maps import METRICS, MAP_LOWER_IS_BETTER
from load_CVIs import choose_index
from utils import reencode, remove_dups


def compute_best_match_analysis_per_dataset(datasets, metrics, labels_folder, file_prefix):
    """
    For each dataset and clustering algorithm:
    1. Find parameterization with highest ARI
    2. Check if that parameterization also has best CVI value
    3. Count correct evaluations and errors

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
        pattern = f"{labels_folder}/labels_{dataset_name}_*.npy"
        label_files = glob.glob(pattern)

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
            algo_name = parts[0] if len(parts) > 0 else "unknown"

            if algo_name not in algo_groups:
                algo_groups[algo_name] = []
            algo_groups[algo_name].append(label_file)

        # Storage for this dataset
        dataset_results = {metric: {'correct': 0, 'errors': 0} for metric in metrics}

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

                # Skip single-cluster results
                if len(np.unique(labels_clustering)) == 1:
                    continue

                labels_clustering_re = reencode(labels_clustering)

                # Compute ARI
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
                    'cvi': cvi_results
                })

            if len(param_results) < 2:
                print(f"    Skipping {algo_name}: insufficient valid parameterizations")
                continue

            # Find parameterization with highest ARI
            best_ari_idx = np.argmax([p['ari'] for p in param_results])
            best_ari_param = param_results[best_ari_idx]

            print(f"    Best ARI: {best_ari_param['ari']:.4f}")

            # For each CVI, check if best ARI also gives best CVI
            for metric in metrics:
                # Collect valid CVI values
                valid_cvi_values = [(i, p['cvi'][metric]) for i, p in enumerate(param_results)
                                    if p['cvi'][metric] is not None]

                if len(valid_cvi_values) < 2:
                    continue

                # Determine best CVI index
                lower_is_better = True if metric.lower() in MAP_LOWER_IS_BETTER else False

                if lower_is_better:
                    best_cvi_idx = min(valid_cvi_values, key=lambda x: x[1])[0]
                else:
                    best_cvi_idx = max(valid_cvi_values, key=lambda x: x[1])[0]

                # Check if best ARI matches best CVI
                if best_ari_idx == best_cvi_idx:
                    dataset_results[metric]['correct'] += 1
                    print(f"      {metric}: CORRECT")
                else:
                    # Count how many parameterizations have better CVI than the best ARI one
                    best_ari_cvi = best_ari_param['cvi'][metric]

                    if best_ari_cvi is None:
                        continue

                    error_count = 0
                    for idx, cvi_val in valid_cvi_values:
                        if idx == best_ari_idx:
                            continue

                        if lower_is_better:
                            if cvi_val < best_ari_cvi:
                                error_count += 1
                        else:
                            if cvi_val > best_ari_cvi:
                                error_count += 1

                    dataset_results[metric]['errors'] += error_count
                    print(f"      {metric}: ERROR (count={error_count})")

            # Store results by algorithm
            if algo_name not in results_by_algo:
                results_by_algo[algo_name] = {metric: {'correct': 0, 'errors': 0} for metric in metrics}

            for metric in metrics:
                # We need to recompute for algo aggregation
                valid_cvi_values = [(i, p['cvi'][metric]) for i, p in enumerate(param_results)
                                    if p['cvi'][metric] is not None]

                if len(valid_cvi_values) < 2:
                    continue

                lower_is_better = True if metric.lower() in MAP_LOWER_IS_BETTER else False

                if lower_is_better:
                    best_cvi_idx = min(valid_cvi_values, key=lambda x: x[1])[0]
                else:
                    best_cvi_idx = max(valid_cvi_values, key=lambda x: x[1])[0]

                if best_ari_idx == best_cvi_idx:
                    results_by_algo[algo_name][metric]['correct'] += 1
                else:
                    best_ari_cvi = best_ari_param['cvi'][metric]
                    if best_ari_cvi is not None:
                        error_count = 0
                        for idx, cvi_val in valid_cvi_values:
                            if idx == best_ari_idx:
                                continue
                            if lower_is_better:
                                if cvi_val < best_ari_cvi:
                                    error_count += 1
                            else:
                                if cvi_val > best_ari_cvi:
                                    error_count += 1
                        results_by_algo[algo_name][metric]['errors'] += error_count

        results_by_dataset[dataset_name] = dataset_results

    return results_by_dataset, results_by_algo


def save_results(results_by_dataset, results_by_algo, file_prefix):
    """Save results as CSV files"""

    # Create DataFrame aggregated by dataset
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
    df_by_algo_pivot = df_by_algo.pivot(index='metric', columns='algorithm',
                                        values=['correct', 'errors'])

    output_path_algo = FOLDER_RESULTS_CORRELATION + f"{file_prefix}_best_match_by_algorithm.csv"
    df_by_algo_pivot.to_csv(output_path_algo)
    print(f">>> Results by algorithm saved to: {output_path_algo}")


def main_real_data():
    from load_datasets import create_real_datasets

    datasets = create_real_datasets()
    prefix = "realdata"

    results_by_dataset, results_by_algo = compute_best_match_analysis_per_dataset(
        datasets=datasets,
        metrics=METRICS,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
        file_prefix=prefix,
    )

    save_results(results_by_dataset, results_by_algo, prefix)


def main_synth_data():
    from load_datasets import create_synthetic_datasets

    datasets = create_synthetic_datasets()
    prefix = "synthdata"

    results_by_dataset, results_by_algo = compute_best_match_analysis_per_dataset(
        datasets=datasets,
        metrics=METRICS,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
        file_prefix=prefix,
    )

    save_results(results_by_dataset, results_by_algo, prefix)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected."
    )

    main_real_data()
    # main_synth_data()