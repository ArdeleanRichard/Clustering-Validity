import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering, estimate_bandwidth, MeanShift, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from constants import scale, FOLDER_RESULTS_CLUSTERING_PARAMS, random_state
from utils import remove_dups, reencode


def get_param_grids():
    """Define parameter grids for each clustering algorithm"""
    param_grids = {
        # 'DBSCAN': {
        #     'eps': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75],
        #     'min_samples': [3, 5, 10, 15, 20, 30, 50, 100, 200]
        # },
        # 'HDBSCAN': {
        #     'min_cluster_size': [2, 3, 5, 10, 15, 20, 30, 50, 100],
        #     'min_samples': [1, 3, 5, 10, 20],
        #     'cluster_selection_epsilon': [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25]
        # },
        # 'MeanShift': {
        #     'quantile': [0.01, 0.02, 0.05, 0.075, 0.1, 0.0125, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
        #     'n_samples': [3, 5, 10, 15, 20],
        #     'bin_seeding': [True, False]
        # },
        # 'AgglomerativeClustering': {
        #     'n_clusters': None,
        #     'linkage': ['ward', 'complete', 'average', 'single']
        # },
        # 'SpectralClustering': {
        #     'n_clusters': None,
        #     'affinity': ['nearest_neighbors', 'rbf'],
        #     'n_neighbors': [3, 5, 10, 15, 20],
        #     'assign_labels': ['kmeans', 'discretize'],
        # },
        # 'KMeans': {
        #     'n_clusters': None,
        #     'max_iter': [100, 300, 500],
        #     'tol': [1e-3, 1e-4, 1e-5],
        #     'algorithm': ["full", "elkan"],  # for newer sklearn versions change 'full' to 'lloyd'
        # },

        'DBSCAN': {
            'eps': [0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
            'min_samples': [3, 5, 10, 20, 50],
        },
        'HDBSCAN': {
            'min_cluster_size': [5, 10, 20],
            'min_samples': [5, 10, 20],
            'cluster_selection_epsilon': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        },
        'MeanShift': {
            # 'quantile': [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
            'quantile': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
            # 'n_samples': [2, 3, 5, 10, 15, 20],
            'n_samples': [2, 3, 5, 10, 20],
            'bin_seeding': [True, False]
        },
        'AgglomerativeClustering': {
            'n_clusters': None,
            'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            'linkage': ['ward', 'complete', 'average', 'single'],
        },
        'SpectralClustering': {
            'n_clusters': None,
            'affinity': ['nearest_neighbors', 'rbf'],
            'n_neighbors': [5, 10, 20],
            'assign_labels': ['kmeans', 'discretize'],
        },
        'KMeans': {
            'n_clusters': None,
            'n_init': 'auto',
            'max_iter': [100, 300, 500],
            'tol': [1e-3, 1e-4, 1e-5],
            'algorithm': ["lloyd", "elkan"], # for newer sklearn versions change 'full' to 'lloyd'
        },

    }
    return param_grids


def evaluate_clustering(labels, true_labels=None):
    """Evaluate clustering quality using multiple metrics"""
    metrics = {}

    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['ami'] = adjusted_mutual_info_score(true_labels, labels)

    valid_mask = labels != -1
    n_clusters = len(set(labels[valid_mask])) if valid_mask.any() else 0
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = (labels == -1).sum()

    return metrics


def grid_search(algo_name, X, true_labels=None, param_grid=None, n_clusters_range=None):
    """Unified grid search function for all clustering algorithms"""

    if param_grid is None:
        param_grid = get_param_grids()[algo_name]

    if n_clusters_range is None:
        n_clusters_range = range(2, 11)

    results = []
    all_labels = []
    best_score = -1
    best_params = None
    best_labels = None

    # Generate parameter combinations based on algorithm
    if algo_name == 'DBSCAN':
        param_combos = [(eps, min_samples)
                        for eps in param_grid['eps']
                        for min_samples in param_grid['min_samples']]

        for eps, min_samples in param_combos:
            start_time = time.time()
            try:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                labels = clusterer.fit_predict(X)
                fit_time = time.time() - start_time

                metrics = evaluate_clustering(labels, true_labels)
                result = {'eps': eps, 'min_samples': min_samples, 'time': fit_time, **metrics}
                results.append(result)
                all_labels.append(labels)

                unique_labels = np.unique(labels)
                if len(unique_labels) > 2:
                    score = metrics['ari']
                else:
                    score = -1

                if np.count_nonzero(labels==-1) > len(X) * 0.4:
                    score = -1

                if score >= best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_labels = labels
            except Exception as e:
                print(f"{algo_name} failed with eps={eps}, min_samples={min_samples}: {e}")

    elif algo_name == 'HDBSCAN':
        param_combos = [(mcs, ms, cse)
                        for mcs in param_grid['min_cluster_size']
                        for ms in param_grid['min_samples']
                        for cse in param_grid['cluster_selection_epsilon']]

        for min_cluster_size, min_samples, cluster_selection_epsilon in param_combos:
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
                all_labels.append(labels)

                unique_labels = np.unique(labels)
                if len(unique_labels) > 2:
                    score = metrics['ari']
                else:
                    score = -1

                if np.count_nonzero(labels==-1) > len(X) * 0.4:
                    score = -1

                if score >= best_score:
                    best_score = score
                    best_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'cluster_selection_epsilon': cluster_selection_epsilon
                    }
                    best_labels = labels
            except Exception as e:
                print(f"{algo_name} failed: {e}")

    elif algo_name == 'MeanShift':
        param_combos = [(q, ns, bs)
                        for q in param_grid['quantile']
                        for ns in param_grid['n_samples']
                        for bs in param_grid['bin_seeding']]

        for quantile, n_samples, bin_seeding in param_combos:
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
                    'quantile': quantile,
                    'n_samples': n_samples,
                    'bandwidth': bandwidth,
                    'bin_seeding': bin_seeding,
                    'time': fit_time,
                    **metrics
                }
                results.append(result)
                all_labels.append(labels)

                score = metrics['ari']
                if score > best_score:
                    best_score = score
                    best_params = {
                        'quantile': quantile,
                        'n_samples': n_samples,
                        'bandwidth': bandwidth,
                        'bin_seeding': bin_seeding
                    }
                    best_labels = labels
            except Exception as e:
                print(f"{algo_name} failed with bandwidth={bandwidth}: {e}")

    elif algo_name == 'AgglomerativeClustering':
        param_combos = [(nc, m, link)
                        for nc in n_clusters_range
                        for m in param_grid['metric']
                        for link in param_grid['linkage']]

        for n_clusters, metric, linkage in param_combos:
            start_time = time.time()
            try:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
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
                all_labels.append(labels)

                score = metrics['ari']
                if score > best_score:
                    best_score = score
                    best_params = {'n_clusters': n_clusters, 'metric': metric, 'linkage': linkage}
                    best_labels = labels
            except Exception as e:
                print(f"{algo_name} failed: {e}")

    elif algo_name == 'SpectralClustering':
        param_combos = [(nc, aff, nn, al)
                        for nc in n_clusters_range
                        for aff in param_grid['affinity']
                        for nn in param_grid['n_neighbors']
                        for al in param_grid['assign_labels']]

        for n_clusters, affinity, n_neighbors, assign_labels in param_combos:
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
                all_labels.append(labels)

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
                print(f"{algo_name} failed: {e}")

    elif algo_name == 'KMeans':
        param_combos = [(nc, mi, tol, algo)
                        for nc in n_clusters_range
                        for mi in param_grid['max_iter']
                        for tol in param_grid['tol']
                        for algo in param_grid['algorithm']]

        for (n_clusters, max_iter, tol, algorithm) in param_combos:
            start_time = time.time()
            try:
                clusterer = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, algorithm=algorithm, random_state=random_state)
                labels = clusterer.fit_predict(X)
                fit_time = time.time() - start_time

                metrics = evaluate_clustering(labels, true_labels)
                result = {
                    'n_clusters': n_clusters,
                    'time': fit_time,
                    **metrics
                }
                results.append(result)
                all_labels.append(labels)

                score = metrics['ari']
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_clusters': n_clusters,
                        'max_iter': max_iter,
                        'tol': tol,
                        'algorithm': algorithm,
                    }
                    best_labels = labels
            except Exception as e:
                print(f"{algo_name} failed: {e}")

    results_df = pd.DataFrame(results)
    return {'params': best_params, 'labels': best_labels, 'score': best_score, 'all_labels': all_labels}, results_df


def run_comprehensive_grid_search(X, true_labels=None, n_clusters_range=None, algorithms=None):
    """Run grid search for all algorithms"""

    # If n_clusters_range not provided, infer from true labels
    if n_clusters_range is None and true_labels is not None:
        n_unique = len(np.unique(true_labels))
        n_clusters_range = range(max(2, n_unique - 2), n_unique + 3)
        print(f"Using inferred n_clusters_range: {list(n_clusters_range)}")
    elif n_clusters_range is None:
        n_clusters_range = range(2, 11)

    if algorithms is None:
        algorithms = ['DBSCAN', 'HDBSCAN', 'MeanShift', 'AgglomerativeClustering', 'SpectralClustering', 'KMeans']

    all_results = {}
    for i, algo_name in enumerate(algorithms, 1):
        print(f"\n[{i}/{len(algorithms)}] Running {algo_name} grid search...")

        best, results_df = grid_search(
            algo_name,
            X,
            true_labels=true_labels,
            n_clusters_range=n_clusters_range
        )

        all_results[algo_name] = {'best': best, 'results': results_df}

        print(f"  Best params: {best['params']}")
        print(f"  Best score: {best['score']:.4f}")
        print()

    return all_results


def compare_best_results(all_results):
    """Create a summary comparison of the best results"""
    summary = []

    for algo_name, results in all_results.items():
        best = results['best']
        labels = best['labels']
        summary.append({
            'Algorithm': algo_name,
            'Best Score': best['score'],
            'N Clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'N Noise': (labels == -1).sum(),
            'Parameters': str(best['params'])
        })

    summary_df = pd.DataFrame(summary).sort_values('Best Score', ascending=False)
    return summary_df


def save_best_parameters(all_results, dataset_name, output_dir=FOLDER_RESULTS_CLUSTERING_PARAMS):
    """Save best parameters and labels for each algorithm"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    labels_path = output_path / 'labels'
    labels_path.mkdir(exist_ok=True)

    all_labels_path = output_path / 'all_labels'
    all_labels_path.mkdir(exist_ok=True)

    best_params = {}

    for algo_name, results in all_results.items():
        params = results['best']['params'].copy()

        # Convert numpy types to native Python types
        for key, value in params.items():
            if isinstance(value, (np.integer, np.floating)):
                params[key] = value.item()
            elif value is None:
                params[key] = None

        best_params[algo_name] = {
            'params': params,
            'score': float(results['best']['score'])
        }

        # Save best labels
        labels = results['best']['labels']
        if labels is not None:
            label_filename = labels_path / f'labels_{dataset_name}_{algo_name}.npy'
            np.save(str(label_filename), labels)
            print(f"Saved {label_filename.name}")

        # Save all labels from all parameter combinations
        all_labels_list = results['best']['all_labels']
        if all_labels_list:
            for idx, labels in enumerate(all_labels_list, start=1):
                all_label_filename = all_labels_path / f'labels_{dataset_name}_{algo_name}_params{idx}.npy'
                np.save(str(all_label_filename), labels)
            print(f"Saved {len(all_labels_list)} parameter combinations for {algo_name}")

    # Load existing parameters or create new dict
    json_file = output_path / 'best_parameters.json'
    if json_file.exists():
        with open(json_file, 'r') as f:
            all_datasets_params = json.load(f)
    else:
        all_datasets_params = {}

    all_datasets_params[dataset_name] = best_params

    with open(json_file, 'w') as f:
        json.dump(all_datasets_params, f, indent=2)

    print(f"Saved all parameter labels to {all_labels_path}/")
    print(f"Saved best parameters to {json_file}")
    print(f"Saved best labels to {labels_path}/")




def main_synthetic_data():
    from load_datasets import create_synthetic_datasets

    datasets = create_synthetic_datasets()

    for data_name, (X, gt) in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_name}")
        print(f"{'=' * 80}")

        X = MinMaxScaler(scale).fit_transform(X)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)


        results = run_comprehensive_grid_search(X, true_labels=gt,
                                                algorithms=[
                                                    "KMeans",
                                                    "DBSCAN",
                                                    "MeanShift",
                                                    "HDBSCAN"
                                                    "AgglomerativeClustering",
                                                    "SpectralClustering"
                                                ])

        save_best_parameters(results, data_name)

        summary = compare_best_results(results)
        print(summary.to_string(index=False))




def main_real_data():
    from load_datasets import create_real_datasets_uci

    datasets = create_real_datasets_uci()

    for data_name, (X, gt) in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_name}")
        print(f"{'=' * 80}")

        X = MinMaxScaler(scale).fit_transform(X)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)

        results = run_comprehensive_grid_search(X, true_labels=gt,
                                                algorithms=[
                                                    "KMeans",
                                                    "DBSCAN",
                                                    "MeanShift",
                                                    "HDBSCAN"
                                                    "AgglomerativeClustering",
                                                    "SpectralClustering"
                                                ])

        save_best_parameters(results, data_name)

        summary = compare_best_results(results)
        print(summary.to_string(index=False))


def main_real_data_new():
    from load_datasets import create_real_datasets_new

    datasets = create_real_datasets_new()

    for data_name, (X, gt) in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_name}")
        print(f"{'=' * 80}")

        X = MinMaxScaler(scale).fit_transform(X)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)


        results = run_comprehensive_grid_search(X, true_labels=gt,
                                                algorithms=[
                                                    # "KMeans",
                                                    # "DBSCAN",
                                                    # "MeanShift",
                                                    "HDBSCAN"
                                                    # "AgglomerativeClustering",
                                                    # "SpectralClustering"
                                                ])

        save_best_parameters(results, data_name)

        summary = compare_best_results(results)
        print(summary.to_string(index=False))

if __name__ == '__main__':
    # main_synthetic_data()
    # main_real_data()
    main_real_data_new()