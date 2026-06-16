import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

from constants import scale, FOLDER_RESULTS_CORRELATION, FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS
from constants_maps import CVIs, MAP_CVI_LOWER_IS_BETTER, MAP_EXTERNAL_CVIs
from cvis_ours.external_CVIs import balanced_external
from load_CVIs import choose_CVI
from main_analysis_cache import _all_caches_exist, _load_cvi_cache, _load_external_cache, _save_cvi_cache, _save_external_cache
from utils import reencode, remove_dups, get_label_files


def compute_single_correlation(metric, external_vals, cvi_vals, dataset_name):
    """Helper function to compute a single Spearman correlation"""
    external_arr = np.array(external_vals)
    cvi_arr = np.array(cvi_vals)

    valid_indices = (
        ~np.isnan(cvi_arr) & np.isfinite(cvi_arr) &
        ~np.isnan(external_arr) & np.isfinite(external_arr)
    )

    if np.sum(valid_indices) < 2:
        print(f"Warning: Insufficient valid data for metric {metric} on {dataset_name}")
        return np.nan

    if np.sum(~valid_indices) > 0:
        print(f"Warning: metric {metric} contains {np.sum(~valid_indices)} NaN/inf values")

    external_valid = external_arr[valid_indices]
    cvi_valid      = cvi_arr[valid_indices]

    corr, p_value = spearmanr(external_valid, cvi_valid)
    print(f"  {metric}: correlation={corr:.3f}, p-value={p_value:.4f}")

    if metric.lower() in MAP_CVI_LOWER_IS_BETTER:
        return -corr
    else:
        return corr


# ---------------------------------------------------------------------------
# Per-clusterer analysis
# ---------------------------------------------------------------------------

def compute_ari_cvi_correlations_per_clusterer(datasets, cvis, labels_folder):
    """
    Compute Spearman correlation between ARI/BARI and internal CVIs for each
    clustering algorithm.  Intermediary CVI and external-metric values are
    cached as CSVs under FOLDER_CVI_CACHE so that subsequent runs skip the
    expensive CVI computation entirely.

    Cache layout (all CSVs share the same column order = param file stems):
      {dataset}_{clusterer}.csv          rows=CVIs,  cols=params  (full)
      {dataset}_{clusterer}_ARI.csv      1 row,      cols=params

    Returns a dict with DataFrames: 'ari',
    where columns are clusterer names and rows are CVI metrics.
    """
    # Structure to collect values per clusterer
    # clusterer_stats[name] = {
    #   'ari': [],
    #   'cvi': {m: []},
    # }
    clusterer_stats = {}

    for dataset_name, (X_raw, labels_gt) in datasets:
        print(f"\n{'-'*40}\nProcessing dataset: {dataset_name} with {X_raw.shape}\n{'-'*40}")

        # Preprocess
        X = MinMaxScaler(scale).fit_transform(X_raw)
        X, labels_gt = remove_dups(X, labels_gt)
        labels_gt_re = reencode(labels_gt)

        # Get label files for this dataset
        pattern = f"{labels_folder}/labels_{dataset_name}_*.npy"
        label_files = get_label_files(pattern, dataset_name)

        if len(label_files) == 0:
            print(f"Warning: No label files found for {dataset_name} with pattern {pattern}")
            continue

        # Group label files by clusterer name
        clusterer_files = {}   # clusterer_name -> [(param_key, label_file), ...]
        for label_file in label_files:
            stem = Path(label_file).stem
            clusterer_name = stem.replace(f"labels_{dataset_name}_", "").split("_")[0]
            param_key = stem  # use the full stem as a unique column identifier
            clusterer_files.setdefault(clusterer_name, []).append((param_key, label_file))

        for clusterer_name, keyed_files in clusterer_files.items():
            # Ensure clusterer entry exists
            if clusterer_name not in clusterer_stats:
                clusterer_stats[clusterer_name] = {
                    'ari': [],
                    'cvi': {m: [] for m in cvis},
                }

            # ---------------------------------------------------------------
            # Try loading from cache
            # ---------------------------------------------------------------
            if _all_caches_exist(dataset_name, clusterer_name):
                print(f"  [cache] Loading {dataset_name} / {clusterer_name} from cache")

                cached = _load_cvi_cache(dataset_name, clusterer_name, suffix="")
                if cached is None:
                    print(f"  [cache] WARNING: cache files exist but failed to load — recomputing")
                else:
                    cvi_dict, param_keys = cached

                    ari_cached    = _load_external_cache(dataset_name, clusterer_name, f"{GROUND_TRUTH_INDEX}")

                    if None in (ari_cached, ):
                        print(f"  [cache] WARNING: some external caches missing — recomputing")
                    else:
                        clusterer_stats[clusterer_name]['ari']    .extend(ari_cached[0])
                        for m in cvis:
                            if m in cvi_dict:
                                clusterer_stats[clusterer_name]['cvi'][m].extend(cvi_dict[m])
                        continue   # skip computation for this (dataset, clusterer) pair

            # ---------------------------------------------------------------
            # Compute from scratch and cache
            # ---------------------------------------------------------------
            print(f"  [compute] {dataset_name} / {clusterer_name} ({len(keyed_files)} params)")

            # Per-(dataset,clusterer) accumulators for caching
            cache_cvi    = {m: [] for m in cvis}
            cache_ari    = []
            param_keys   = []

            for label_id, (param_key, label_file) in enumerate(keyed_files):
                print(f"\tLabel files: {label_id+1}/{len(keyed_files)}")

                try:
                    labels_clustering = np.load(label_file)
                except Exception as e:
                    print(f"Warning: Failed to load {label_file}: {e}")
                    continue

                # skip trivial clustering
                if len(np.unique(labels_clustering)) == 1:
                    continue

                labels_clustering_re = reencode(labels_clustering)


                # --- ARI ---
                try:
                    ari_val = gti_function(labels_gt_re, labels_clustering_re)
                except Exception as e:
                    print(f"Warning: ARI failed for {clusterer_name} on {dataset_name}: {e}")
                    ari_val = np.nan
                clusterer_stats[clusterer_name]['ari'].append(ari_val)
                cache_ari.append(ari_val)

                # --- CVIs ---
                for metric_id, metric in enumerate(cvis):
                    start = time.time()

                    # full
                    try:
                        c_full = choose_CVI(cvi=metric, data=X, labels=labels_clustering)
                    except Exception:
                        c_full = np.nan
                    print(f"\t\tMetric - {metric}: {metric_id + 1}/{len(cvis)} in {time.time() - start:.3f}s")
                    clusterer_stats[clusterer_name]['cvi'][metric].append(c_full)
                    cache_cvi[metric].append(c_full)

                param_keys.append(param_key)

            # Save to cache (only if we processed at least one parameterisation)
            if len(param_keys) > 0:
                _save_cvi_cache(dataset_name, clusterer_name, cache_cvi, param_keys, suffix="")
                _save_external_cache(dataset_name, clusterer_name, f"_{GROUND_TRUTH_INDEX}",     cache_ari,     param_keys)
                print(f"  [cache] Saved cache for {dataset_name} / {clusterer_name}")

    # -----------------------------------------------------------------------
    # Compute correlations per clusterer
    # -----------------------------------------------------------------------
    results_ari     = {}

    for clusterer_name, stats in clusterer_stats.items():
        print(f"\nComputing correlations for clusterer: {clusterer_name}")

        corr_ari     = {}

        for metric in cvis:
            corr_ari[metric]     = compute_single_correlation(metric, stats['ari'],     stats['cvi'][metric],    clusterer_name)

        results_ari[clusterer_name]     = corr_ari

    return {
        'ari':     pd.DataFrame(results_ari),
    }


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def compute_ari_cvi_correlations_per_dataset(datasets, metrics, labels_folder):
    """
    Compute Spearman correlation between ARI and internal CVIs for each
    dataset.  Intermediary results are cached per (dataset, clusterer) using
    the same CSV layout as compute_ari_cvi_correlations_per_clusterer.

    Returns a dict with DataFrames: 'ari',
    where columns are dataset names and rows are CVI metrics.
    """
    results_ari     = {}

    for data_id, (dataset_name, (X, labels_gt)) in enumerate(datasets):
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name}: {data_id+1}/{len(datasets)}")
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

        # Group by clusterer so we can cache per (dataset, clusterer)
        clusterer_files = {}
        for label_file in label_files:
            stem = Path(label_file).stem
            clusterer_name = stem.replace(f"labels_{dataset_name}_", "").split("_")[0]
            param_key = stem
            clusterer_files.setdefault(clusterer_name, []).append((param_key, label_file))

        # Accumulators across all clusterers for this dataset
        ari_values    = []
        cvi_values    = {metric: [] for metric in metrics}

        for clusterer_name, keyed_files in clusterer_files.items():
            # -----------------------------------------------------------
            # Try loading from cache
            # -----------------------------------------------------------
            if _all_caches_exist(dataset_name, clusterer_name):
                print(f"  [cache] Loading {dataset_name} / {clusterer_name} from cache")

                cached = _load_cvi_cache(dataset_name, clusterer_name, suffix="")
                ari_cached     = _load_external_cache(dataset_name, clusterer_name, f"_{GROUND_TRUTH_INDEX}")

                if None in (cached, ari_cached, ):
                    print(f"  [cache] WARNING: cache load failed for {clusterer_name} — recomputing")
                else:
                    cvi_dict, _ = cached
                    ari_values    .extend(ari_cached[0])

                    for m in metrics:
                        if m in cvi_dict:
                            cvi_values[m]   .extend(cvi_dict[m])
                    continue   # skip computation

            # -----------------------------------------------------------
            # Compute from scratch and cache
            # -----------------------------------------------------------
            print(f"  [compute] {dataset_name} / {clusterer_name} ({len(keyed_files)} params)")

            cache_cvi    = {m: [] for m in metrics}
            cache_ari    = []
            param_keys   = []

            for label_id, (param_key, label_file) in enumerate(keyed_files):
                print(f"\tLabel files: {label_id + 1}/{len(keyed_files)}")

                try:
                    labels_clustering = np.load(label_file)
                except Exception as e:
                    print(f"Warning: Failed to load {label_file}: {e}")
                    continue

                if len(np.unique(labels_clustering)) == 1:
                    continue
                if X.shape[0] != labels_clustering.shape[0]:
                    continue

                labels_clustering_re  = reencode(labels_clustering)


                # ARI
                ari_val    = gti_function(labels_gt_re, labels_clustering_re)
                ari_values   .append(ari_val)
                cache_ari    .append(ari_val)

                # CVIs
                for metric_id, metric in enumerate(metrics):
                    start = time.time()
                    try:
                        c_full = choose_CVI(cvi=metric, data=X, labels=labels_clustering)
                    except Exception as e:
                        c_full = np.nan

                    cvi_values[metric]   .append(c_full)
                    cache_cvi[metric]    .append(c_full)
                    print(f"\t\tCVI - {metric}: {metric_id + 1}/{len(metrics)} in {time.time() - start:.3f}s")

                param_keys.append(param_key)

            # Save cache for this (dataset, clusterer) pair
            if len(param_keys) > 0:
                _save_cvi_cache(dataset_name, clusterer_name, cache_cvi, param_keys, suffix="")
                _save_external_cache(dataset_name, clusterer_name, f"_{GROUND_TRUTH_INDEX}",     cache_ari,     param_keys)
                print(f"  [cache] Saved cache for {dataset_name} / {clusterer_name}")

        # Compute correlations for this dataset (unchanged logic)
        dataset_correlations_ari     = {}

        for metric in metrics:
            dataset_correlations_ari[metric]     = compute_single_correlation(metric, ari_values,     cvi_values[metric],    dataset_name)

        results_ari[dataset_name]     = dataset_correlations_ari

    return {
        'ari':     pd.DataFrame(results_ari),
    }



def save_correlation_matrix(df, file_name="correlations_cvi_to_ari_per_dataset"):
    """Save correlation matrix to CSV"""
    output_path = FOLDER_RESULTS_CORRELATION + f"{file_name}.csv"
    df.to_csv(output_path)
    print(f"\n>>> Results saved to: {output_path}")
    print(f">>> Shape: {df.shape[0]} CVIs x {df.shape[1]} datasets")



def main_analysis_per_dataset(data_type):
    cvis = CVIs.copy()
    cvis.remove("ED-S")
    cvis.remove("ED-DB")
    cvis.remove("ED-CH")

    if data_type == "data_synth":
        from load_datasets import create_synthetic_datasets
        datasets = create_synthetic_datasets()

    if data_type == "data_real":
        cvis.remove("CDbw")  # cannot construct hull
        from load_datasets import create_real_datasets_uci
        datasets = create_real_datasets_uci()

    if data_type == "data_image":
        from load_datasets import create_real_datasets_image
        datasets = create_real_datasets_image(image_size=(64, 64))

        cvis.remove("CDbw")  # cannot construct hull
        cvis.remove("rCIP")  # Failed to compute


    correlation_matrices = compute_ari_cvi_correlations_per_dataset(
        datasets=datasets,
        metrics=cvis,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
    )

    save_correlation_matrix(correlation_matrices['ari'],     file_name=f"{data_type}_correlations_per_dataset_cvi_to_ari")


def main_analysis_per_clusterer(data_type):
    cvis = CVIs.copy()
    cvis.remove("ED-S")
    cvis.remove("ED-DB")
    cvis.remove("ED-CH")

    if data_type == "data_synth":
        from load_datasets import create_synthetic_datasets
        datasets = create_synthetic_datasets()

    if data_type == "data_real":
        cvis.remove("CDbw")  # cannot construct hull
        from load_datasets import create_real_datasets_uci
        datasets = create_real_datasets_uci()

    if data_type == "data_image":
        from load_datasets import create_real_datasets_image
        datasets = create_real_datasets_image(image_size=(64, 64))

        cvis.remove("CDbw")  # cannot construct hull
        cvis.remove("rCIP")  # Failed to compute

    correlation_matrices = compute_ari_cvi_correlations_per_clusterer(
        datasets=datasets,
        cvis=cvis,
        labels_folder=FOLDER_RESULTS_CLUSTERING_LABELS_ALL_PARAMETERS,
    )

    save_correlation_matrix(correlation_matrices['ari'],     file_name=f"{data_type}_correlations_per_clusterer_cvi_to_ari")


if __name__ == "__main__":
    GROUND_TRUTH_INDEX = "ARI"
    _, _, gti_function = MAP_EXTERNAL_CVIs[GROUND_TRUTH_INDEX.lower()]

    main_analysis_per_dataset(data_type="data_synth")
    main_analysis_per_clusterer(data_type="data_synth")

    main_analysis_per_dataset(data_type="data_real")
    main_analysis_per_clusterer(data_type="data_real")

    main_analysis_per_dataset(data_type="data_image")
    main_analysis_per_clusterer(data_type="data_image")