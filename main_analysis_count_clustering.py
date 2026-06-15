import os
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import math

from constants import scale, FOLDER_RESULTS_COUNT, FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS, LABEL_COLOR_MAP
from constants_maps import CVIs, MAP_CVI_LOWER_IS_BETTER
from load_CVIs import choose_CVI
from main_analysis_cache import _main_caches_exist, _load_external_cache, _load_cvi_cache, _save_cvi_cache, _save_external_cache
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




def compute_count_analysis_per_dataset(datasets, cvis, labels_folder, create_plots=True, plot_output_folder=None):
    """
    For each dataset and clustering algorithm:
    1. Find parameterization with highest ARI
    2. Check if that parameterization also has best CVI value
    3. Count correct evaluations (binary: 1 if match, 0 otherwise)
    4. Count errors (number of parameterizations with better CVI + failed ones)
    5. Create scatter plots for erroneous cases

    Intermediary CVI and ARI values are cached as CSVs under FOLDER_CVI_CACHE
    so that subsequent runs can skip the expensive CVI computation entirely.

    Cache layout per (dataset, algo) pair:
      {dataset}_{algo}.csv      rows=CVIs,   cols=param-file-stems
      {dataset}_{algo}_ARI.csv  1 row,       cols=param-file-stems  (ARI values)

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
            parts = filename.replace(f"labels_{dataset_name}_", "").split("_")
            algo_name = parts[-2] if len(parts) > 1 else "unknown"

            if algo_name not in algo_groups:
                algo_groups[algo_name] = []
            algo_groups[algo_name].append(label_file)

        # Storage for this dataset
        if dataset_name not in results_by_dataset:
            results_by_dataset[dataset_name] = {metric: {'correct': 0, 'errors': 0} for metric in cvis}

        # Process each clustering algorithm
        for algo_name, algo_files in algo_groups.items():
            print(f"\n  Processing algorithm: {algo_name} ({len(algo_files)} parameterizations)")

            # ------------------------------------------------------------------
            # Attempt to load param_results from cache
            # ------------------------------------------------------------------
            param_results = None   # will be a list of dicts once populated

            if _main_caches_exist(dataset_name, algo_name):
                print(f"  [cache] Loading {dataset_name} / {algo_name} from cache")

                cvi_cached = _load_cvi_cache(dataset_name, algo_name)
                ari_cached = _load_external_cache(dataset_name, algo_name, f"_{GROUND_TRUTH_INDEX}")

                if cvi_cached is None or ari_cached is None:
                    print(f"  [cache] WARNING: cache load failed — recomputing")
                else:
                    cvi_dict, param_keys = cvi_cached
                    ari_values, _ = ari_cached

                    # Reconstruct param_results list (labels not needed for the
                    # analysis logic below — only cvi and ari values are used
                    # after this point, except for scatter plots which need labels)
                    param_results = []
                    for i, (param_key, ari_val) in enumerate(zip(param_keys, ari_values)):
                        cvi_for_param = {
                            m: (cvi_dict[m][i] if m in cvi_dict else None)
                            for m in cvis
                        }
                        # Replace NaN/inf sentinels with None (matches original logic)
                        cvi_for_param = {
                            m: (None if (v is not None and isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                            for m, v in cvi_for_param.items()
                        }
                        param_results.append({
                            'file': param_key,
                            'ari': ari_val,
                            'labels': None,   # not stored in cache; plots disabled for cached runs
                            'cvi': cvi_for_param,
                        })

            # ------------------------------------------------------------------
            # Compute from scratch if cache was missing / unreadable
            # ------------------------------------------------------------------
            if param_results is None:
                param_results = []
                cache_cvi  = {m: [] for m in cvis}
                cache_ari  = []
                param_keys = []

                for label_file in algo_files:
                    try:
                        labels_clustering = np.load(label_file)
                    except Exception as e:
                        print(f"    Warning: Failed to load {label_file}: {e}")
                        continue

                    # Skip single-cluster / all-noise results
                    unique_labels = np.unique(labels_clustering)
                    if len(unique_labels) == 1 or (-1 in unique_labels and len(unique_labels) <= 2):
                        continue

                    labels_clustering_re = reencode(labels_clustering)

                    if len(labels_clustering_re) != len(labels_gt_re):
                        continue

                    ari_value = adjusted_rand_score(labels_gt_re, labels_clustering_re)

                    # Compute all CVIs
                    cvi_results = {}
                    for metric in cvis:
                        try:
                            cvi_value = choose_CVI(cvi=metric, data=X, labels=labels_clustering)
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

                    # Accumulate for cache
                    cache_ari.append(ari_value)
                    param_keys.append(Path(label_file).stem)
                    for m in cvis:
                        # Store None as NaN in the CSV
                        raw = cvi_results[m]
                        cache_cvi[m].append(np.nan if raw is None else raw)

                # Save cache
                if len(param_keys) > 0:
                    _save_cvi_cache(dataset_name, algo_name, cache_cvi, param_keys)
                    _save_external_cache(dataset_name, algo_name, f"_{GROUND_TRUTH_INDEX}", cache_ari, param_keys)
                    print(f"  [cache] Saved cache for {dataset_name} / {algo_name}")

            # ------------------------------------------------------------------
            # Analysis logic
            # ------------------------------------------------------------------
            if len(param_results) < 2:
                print(f"    Skipping {algo_name}: insufficient valid parameterizations")
                continue

            # Find all parameterizations that share the highest ARI
            max_ari = max(p['ari'] for p in param_results)
            best_ari_indices = {i for i, p in enumerate(param_results) if p['ari'] == max_ari}

            print(f"    Best ARI: {max_ari:.4f} ({len(best_ari_indices)} parameterization(s))")

            # Initialize results for this algorithm if not exists
            if algo_name not in results_by_algo:
                results_by_algo[algo_name] = {metric: {'correct': 0, 'errors': 0} for metric in cvis}

            # For each CVI, check if best ARI also gives best CVI
            for metric in cvis:
                # Collect valid CVI values (non-None, i.e. not nan/inf at compute time)
                valid_cvi_values = [(i, p['cvi'][metric]) for i, p in enumerate(param_results) if p['cvi'][metric] is not None]

                # Determine direction first — needed for the edge-case checks below
                lower_is_better = True if metric.lower() in MAP_CVI_LOWER_IS_BETTER else False

                # Edge-case: treat all-zero values as failed (no meaningful signal),
                # but only for higher-is-better CVIs.  For lower-is-better CVIs, 0 can
                # be a legitimate best-possible score, so all-zeros there is fine.
                # This covers two sub-cases for higher-is-better metrics:
                #   (a) every parameterization returned 0  — all zeros
                #   (b) some returned 0 and the rest are None — mix of zeros and failures
                # In both situations the CVI provides no useful information, so the
                # entire (dataset, algo, metric) group is counted as errors.
                if (not lower_is_better
                        and len(valid_cvi_values) > 0
                        and all(v == 0.0 for _, v in valid_cvi_values)):
                    failed_count = len(param_results)
                    results_by_dataset[dataset_name][metric]['errors'] += failed_count
                    results_by_algo[algo_name][metric]['errors'] += failed_count
                    print(f"      {metric}: ERROR — all valid values are 0 (count={failed_count})")
                    continue

                if len(valid_cvi_values) < 2:
                    # Not enough valid values - count failed ones as errors
                    failed_count = len(param_results) - len(valid_cvi_values)
                    results_by_dataset[dataset_name][metric]['errors'] += failed_count
                    results_by_algo[algo_name][metric]['errors'] += failed_count
                    print(f"      {metric}: ERROR (count={failed_count})")
                    continue

                # Collect all indices tied for best CVI value
                if lower_is_better:
                    best_cvi_val = min(v for _, v in valid_cvi_values)
                else:
                    best_cvi_val = max(v for _, v in valid_cvi_values)
                best_cvi_indices = {i for i, v in valid_cvi_values if v == best_cvi_val}

                # Among the best-ARI parameterizations, pick the one with the best CVI
                # value for this metric (i.e. give the best-ARI group every benefit of
                # the doubt).  Fall back to any best-ARI index if none have a valid CVI.
                best_ari_valid = [(i, param_results[i]['cvi'][metric])
                                  for i in best_ari_indices
                                  if param_results[i]['cvi'][metric] is not None]
                if best_ari_valid:
                    if lower_is_better:
                        best_ari_idx = min(best_ari_valid, key=lambda x: x[1])[0]
                    else:
                        best_ari_idx = max(best_ari_valid, key=lambda x: x[1])[0]
                else:
                    # No valid CVI among best-ARI params — pick any for reference
                    best_ari_idx = next(iter(best_ari_indices))
                best_ari_param = param_results[best_ari_idx]

                # Correct if any best-ARI param is also among the best-CVI params
                if best_ari_indices & best_cvi_indices:
                    results_by_dataset[dataset_name][metric]['correct'] += 1
                    results_by_algo[algo_name][metric]['correct'] += 1
                    print(f"      {metric}: CORRECT")
                else:
                    best_ari_cvi = best_ari_param['cvi'][metric]

                    error_cases = []

                    if best_ari_cvi is None:
                        error_count = len(param_results)
                        error_cases = [p for i, p in enumerate(param_results) if i not in best_ari_indices]
                    else:
                        error_count = 0
                        for idx, cvi_val in valid_cvi_values:
                            if idx in best_ari_indices:
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
                    # NOTE: scatter plots require the actual cluster labels, which
                    # are not stored in the cache.  Plots are only produced when
                    # param_results were computed fresh in this run.
                    if create_plots and plot_output_folder and len(error_cases) > 0:
                        # Check whether labels are available (None when loaded from cache)
                        if best_ari_param['labels'] is not None:
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
                        else:
                            print(f"      (scatter plot skipped: labels not available in cache)")

    return results_by_dataset, results_by_algo




def save_results(cvis, results_by_dataset, results_by_algo, file_prefix):
    dataset_rows = []
    for dataset, cvis_dict in results_by_dataset.items():
        for cvi, counts in cvis_dict.items():
            dataset_rows.append({
                'dataset': dataset,
                'cvi': cvi,
                'correct': counts['correct'],
                'errors': counts['errors']
            })

    df_by_dataset = pd.DataFrame(dataset_rows)
    df_by_dataset_pivot = df_by_dataset.pivot(index='cvi', columns='dataset', values=['correct', 'errors'])
    df_by_dataset_pivot = df_by_dataset_pivot.reindex(cvis)  # <-- preserve order

    output_path_dataset = FOLDER_RESULTS_COUNT + f"{file_prefix}_count_by_dataset.csv"
    df_by_dataset_pivot.to_csv(output_path_dataset)
    print(f"\n>>> Results by dataset saved to: {output_path_dataset}")

    # Create DataFrame aggregated by clustering algorithm
    clusterer_rows = []
    for clusterer, cvis_dict in results_by_algo.items():
        for cvi, counts in cvis_dict.items():
            clusterer_rows.append({
                'clusterer': clusterer,
                'cvi': cvi,
                'correct': counts['correct'],
                'errors': counts['errors']
            })

    df_by_clusterer = pd.DataFrame(clusterer_rows)
    df_by_clusterer_pivot = df_by_clusterer.pivot(index='cvi', columns='clusterer', values=['correct', 'errors'])
    df_by_clusterer_pivot = df_by_clusterer_pivot.reindex(cvis)  # <-- preserve order

    output_path_clusterer = FOLDER_RESULTS_COUNT + f"{file_prefix}_count_by_clusterer.csv"
    df_by_clusterer_pivot.to_csv(output_path_clusterer)
    print(f">>> Results by clusterer saved to: {output_path_clusterer}")





def main(data_type):
    cvis = CVIs.copy()
    cvis.remove("ED-S")
    cvis.remove("ED-DB")
    cvis.remove("ED-CH")

    if data_type == "data_synth":
        from load_datasets import create_synthetic_datasets
        datasets = create_synthetic_datasets()

    if data_type == "data_real":
        cvis.remove("CDbw")
        from load_datasets import create_real_datasets_uci
        datasets = create_real_datasets_uci()

    if data_type == "data_image":
        cvis.remove("CDbw")  # cannot construct hull
        cvis.remove("rCIP")  # Failed to compute

        from load_datasets import create_real_datasets_image
        datasets = create_real_datasets_image()


    plot_folder = FOLDER_RESULTS_COUNT + f"error_plots_{data_type}/"


    results_by_dataset, results_by_algo = compute_count_analysis_per_dataset(
        datasets=datasets,
        cvis=cvis,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
        create_plots=True,
        plot_output_folder=plot_folder
    )

    save_results(cvis, results_by_dataset, results_by_algo, data_type)


if __name__ == "__main__":
    GROUND_TRUTH_INDEX = "AMI"

    main("data_synth")
    main("data_real")
    main("data_image")