import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering, estimate_bandwidth, MeanShift, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from typing import Dict, List, Tuple, Any

from constants import scale, FOLDER_RESULTS_CLUSTERING_PARAMS


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Define parameter grids for each clustering algorithm"""
    param_grids = {
        'DBSCAN': {
            'eps': [0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25],
            'min_samples': [3, 5, 10, 15, 20]
        },
        'HDBSCAN': {
            'min_cluster_size': [3, 5, 10, 15, 20, 30],
            'min_samples': [1, 3, 5, 10],
            'cluster_selection_epsilon': [0, 0, 0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25]
        },
        'MeanShift': {
            'quantile': [0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
            'n_samples': [3, 5, 10, 15, 20],
            # 'bandwidth': [0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
            'bin_seeding': [True, False]
        },
        'AgglomerativeClustering': {
            'n_clusters': None,  # Will be set based on dataset
            'linkage': ['ward', 'complete', 'average', 'single']
        },
        'SpectralClustering': {
            'n_clusters': None,  # Will be set based on dataset
            'affinity': ['nearest_neighbors', 'rbf'],
            'n_neighbors': [3, 5, 10, 15, 20],
            'assign_labels': ['kmeans', 'discretize']
        }
    }
    return param_grids


def evaluate_clustering(labels, true_labels=None) -> Dict[str, float]:
    """Evaluate clustering quality using multiple metrics"""
    metrics = {}

    # Filter out noise points (label -1)
    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask])) if valid_mask.any() else 0

    # External metrics (if true labels provided)
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['ami'] = adjusted_mutual_info_score(true_labels, labels)

    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = (labels == -1).sum()

    return metrics


def grid_search_dbscan(X, true_labels=None, param_grid=None) -> Tuple[Dict, pd.DataFrame]:
    """Grid search for DBSCAN"""
    if param_grid is None:
        param_grid = get_param_grids()['DBSCAN']

    results = []
    best_score = -1
    best_params = None
    best_labels = None

    for eps, min_samples in product(param_grid['eps'], param_grid['min_samples']):
        start_time = time.time()

        try:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X)
            fit_time = time.time() - start_time

            metrics = evaluate_clustering(labels, true_labels)

            result = {
                'eps': eps,
                'min_samples': min_samples,
                'time': fit_time,
                **metrics
            }
            results.append(result)

            score = metrics['ari']
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels
        except Exception as e:
            print(f"DBSCAN failed with eps={eps}, min_samples={min_samples}: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score}, results_df


def grid_search_hdbscan(X, true_labels=None, param_grid=None) -> Tuple[Dict, pd.DataFrame]:
    """Grid search for HDBSCAN"""
    if param_grid is None:
        param_grid = get_param_grids()['HDBSCAN']

    results = []
    best_score = -1
    best_params = None
    best_labels = None

    for min_cluster_size, min_samples, cluster_selection_epsilon in product(
            param_grid['min_cluster_size'],
            param_grid['min_samples'],
            param_grid['cluster_selection_epsilon']
    ):
        start_time = time.time()

        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            labels = clusterer.fit_predict(X)
            fit_time = time.time() - start_time

            metrics = evaluate_clustering(labels, true_labels)

            result = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': cluster_selection_epsilon,
                'time': fit_time,
                **metrics
            }
            results.append(result)

            score = metrics['ari']
            if score > best_score:
                best_score = score
                best_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'cluster_selection_epsilon': cluster_selection_epsilon
                }
                best_labels = labels
        except Exception as e:
            print(f"HDBSCAN failed: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score}, results_df


def grid_search_meanshift(X, true_labels=None, param_grid=None) -> Tuple[Dict, pd.DataFrame]:
    """Grid search for MeanShift"""
    if param_grid is None:
        param_grid = get_param_grids()['MeanShift']

    results = []
    best_score = -1
    best_params = None
    best_labels = None

    # for bandwidth, bin_seeding in product(param_grid['bandwidth'], param_grid['bin_seeding']):
    for quantile, n_samples, bin_seeding in product(param_grid['quantile'], param_grid['n_samples'], param_grid['bin_seeding']):
        start_time = time.time()

        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples)

        if bandwidth < 1e-2:
            continue

        try:
            clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
            labels = clusterer.fit_predict(X)
            fit_time = time.time() - start_time

            metrics = evaluate_clustering(labels, true_labels)

            result = {
                'bandwidth': bandwidth,
                'bin_seeding': bin_seeding,
                'time': fit_time,
                **metrics
            }
            results.append(result)

            score = metrics['ari']
            if score > best_score:
                best_score = score
                best_params = {'quantile': quantile, 'n_samples': n_samples, 'bandwidth': bandwidth, 'bin_seeding': bin_seeding}
                best_labels = labels
        except Exception as e:
            print(f"MeanShift failed with bandwidth={bandwidth}: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score}, results_df


def grid_search_agglomerative(X, n_clusters_range, true_labels=None, param_grid=None) -> Tuple[Dict, pd.DataFrame]:
    """Grid search for Agglomerative Clustering"""
    if param_grid is None:
        param_grid = get_param_grids()['AgglomerativeClustering']

    results = []
    best_score = -1
    best_params = None
    best_labels = None

    for n_clusters, linkage in product(n_clusters_range, param_grid['linkage']):
        start_time = time.time()

        try:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = clusterer.fit_predict(X)
            fit_time = time.time() - start_time

            metrics = evaluate_clustering(labels, true_labels)

            result = {
                'n_clusters': n_clusters,
                'linkage': linkage,
                'time': fit_time,
                **metrics
            }
            results.append(result)

            score = metrics['ari']
            if score > best_score:
                best_score = score
                best_params = {'n_clusters': n_clusters, 'linkage': linkage}
                best_labels = labels
        except Exception as e:
            print(f"Agglomerative failed with n_clusters={n_clusters}, linkage={linkage}: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score}, results_df


def grid_search_spectral(X, n_clusters_range, true_labels=None, param_grid=None) -> Tuple[Dict, pd.DataFrame]:
    """Grid search for Spectral Clustering"""
    if param_grid is None:
        param_grid = get_param_grids()['SpectralClustering']

    results = []
    best_score = -1
    best_params = None
    best_labels = None

    for n_clusters, affinity, n_neighbors, assign_labels in product(
            n_clusters_range,
            param_grid['affinity'],
            param_grid['n_neighbors'],
            param_grid['assign_labels']
    ):
        start_time = time.time()

        try:
            if affinity == 'rbf':
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    assign_labels=assign_labels,
                    random_state=0
                )
            else:
                clusterer = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    n_neighbors=n_neighbors,
                    assign_labels=assign_labels,
                    random_state=0
                )
            labels = clusterer.fit_predict(X)
            fit_time = time.time() - start_time

            metrics = evaluate_clustering(labels, true_labels)

            result = {
                'n_clusters': n_clusters,
                'affinity': affinity,
                'n_neighbors': n_neighbors if affinity != 'rbf' else None,
                'assign_labels': assign_labels,
                'time': fit_time,
                **metrics
            }
            results.append(result)

            score = metrics['ari']
            if score > best_score:
                best_score = score
                best_params = {
                    'n_clusters': n_clusters,
                    'affinity': affinity,
                    'n_neighbors': n_neighbors if affinity != 'rbf' else None,
                    'assign_labels': assign_labels
                }
                best_labels = labels
        except Exception as e:
            print(f"Spectral failed: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score}, results_df


def run_comprehensive_grid_search(X, true_labels=None, n_clusters_range=None):
    """Run grid search for all algorithms"""

    if n_clusters_range is None:
        n_clusters_range = range(2, 11)

    all_results = {}

    # DBSCAN
    print("\n[1/5] Running DBSCAN grid search...")
    dbscan_best, dbscan_results = grid_search_dbscan(X, true_labels)
    all_results['DBSCAN'] = {'best': dbscan_best, 'results': dbscan_results}
    print(f"  Best params: {dbscan_best['params']}")
    print(f"  Best score: {dbscan_best['score']:.4f}")

    # HDBSCAN
    print("\n[2/5] Running HDBSCAN grid search...")
    hdbscan_best, hdbscan_results = grid_search_hdbscan(X, true_labels)
    all_results['HDBSCAN'] = {'best': hdbscan_best, 'results': hdbscan_results}
    print(f"  Best params: {hdbscan_best['params']}")
    print(f"  Best score: {hdbscan_best['score']:.4f}")

    # MeanShift
    print("\n[3/5] Running MeanShift grid search...")
    meanshift_best, meanshift_results = grid_search_meanshift(X, true_labels)
    all_results['MeanShift'] = {'best': meanshift_best, 'results': meanshift_results}
    print(f"  Best params: {meanshift_best['params']}")
    print(f"  Best score: {meanshift_best['score']:.4f}")

    # Agglomerative
    print("\n[4/5] Running Agglomerative Clustering grid search...")
    agglom_best, agglom_results = grid_search_agglomerative(X, n_clusters_range, true_labels)
    all_results['AgglomerativeClustering'] = {'best': agglom_best, 'results': agglom_results}
    print(f"  Best params: {agglom_best['params']}")
    print(f"  Best score: {agglom_best['score']:.4f}")

    # Spectral
    print("\n[5/5] Running Spectral Clustering grid search...")
    spectral_best, spectral_results = grid_search_spectral(X, n_clusters_range, true_labels)
    all_results['SpectralClustering'] = {'best': spectral_best, 'results': spectral_results}
    print(f"  Best params: {spectral_best['params']}")
    print(f"  Best score: {spectral_best['score']:.4f}")

    print("\n" + "=" * 80)
    print("Grid Search Complete!")
    print("=" * 80)

    return all_results


def compare_best_results(all_results):
    """Create a summary comparison of the best results"""
    summary = []

    for algo_name, results in all_results.items():
        best = results['best']
        summary.append({
            'Algorithm': algo_name,
            'Best Score': best['score'],
            'N Clusters': len(set(best['labels'])) - (1 if -1 in best['labels'] else 0),
            'N Noise': (best['labels'] == -1).sum(),
            'Parameters': str(best['params'])
        })

    summary_df = pd.DataFrame(summary).sort_values('Best Score', ascending=False)
    return summary_df


def save_best_parameters(all_results, dataset_name, output_dir=FOLDER_RESULTS_CLUSTERING_PARAMS):
    """
    Save the best parameters for each algorithm to a JSON file organized by dataset.

    Args:
        all_results: Dictionary containing grid search results for all algorithms
        dataset_name: Name of the dataset
        output_dir: Directory to save the parameter files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)


    best_params = {}
    for algo_name, results in all_results.items():
        # Convert numpy types to native Python types for JSON serialization
        params = results['best']['params'].copy()
        for key, value in params.items():
            if isinstance(value, (np.integer, np.floating)):
                params[key] = value.item()
            elif value is None:
                params[key] = None

        best_params[algo_name] = {
            'params': params,
            'score': float(results['best']['score'])
        }

    # Load existing parameters file if it exists
    json_file = output_path / f'best_parameters.json'
    if json_file.exists():
        f = open(json_file, 'r')
        all_datasets_params = json.load(f)
        f.close()
    else:
        all_datasets_params = {}

    # Update with new dataset parameters
    all_datasets_params[dataset_name] = best_params

    # Save back to file
    f = open(json_file, 'w')
    json.dump(all_datasets_params, f, indent=2)
    f.close()

    print(f"\n✓ Best parameters saved to {json_file}")



if __name__ == '__main__':
    from load_datasets import create_compound, create_aggregation, create_jain, create_unbalance, create_spiral, create_pathbased
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding may not work as expected."
    )

    datasets = [
        ("compound",        create_compound()),
        ("aggregation",     create_aggregation()),
        ("jain",            create_jain()),
        ("spiral",          create_spiral()),
        ("pathbased",       create_pathbased()),
        ("unbalance",       create_unbalance()),
    ]

    # Run grid search and save parameters for each dataset
    for data_name, (X, gt) in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_name}")
        print(f"{'=' * 80}")

        X = MinMaxScaler(scale).fit_transform(X)

        results = run_comprehensive_grid_search(
            X,
            true_labels=gt,
            n_clusters_range=range(2, 6)
        )

        # Save best parameters for this dataset
        save_best_parameters(results, data_name)

        summary = compare_best_results(results)
        print("\n### Summary of Best Results ###")
        print(summary.to_string(index=False))

        # Access detailed results for each algorithm
        # print("\n### Top 5 DBSCAN configurations ###")
        # print(results['DBSCAN']['results'].nlargest(5, 'ari')[['eps', 'min_samples', 'ari', 'n_clusters']])
