import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, estimate_bandwidth, MeanShift, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from load_datasets import create_set1, create_compound, create_aggregation, create_jain, create_unbalance, create_spiral, create_pathbased
from constants import LABEL_COLOR_MAP, random_state, scale, FOLDER_FIGS_CLUSTERING, FOLDER_RESULTS_CLUSTERING_PARAMS
from cvis_ours.np_kmeans import KMeansClustering
from cvis_ours.ed_kmeans import ED_KMeansClustering
from cvis_ours.ad_kmeans import AD_KMeansClustering

def load_best_parameters_for_dataset(dataset_name):
    from pathlib import Path
    import json

    params_path = Path(FOLDER_RESULTS_CLUSTERING_PARAMS+f'best_parameters.json')
    f = open(params_path, 'r')
    all_params = json.load(f)

    if dataset_name not in all_params:
        return None

    return all_params[dataset_name]


def run_clustering_algorithms(dataset_name, X, n_clusters):
    best_params = load_best_parameters_for_dataset(dataset_name)

    km_start = time.time()
    # Kmeans = KMeansClustering(X, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    km_labels = kmeans.labels_
    km_time = time.time() - km_start

    dbs_start = time.time()
    if 'DBSCAN' not in best_params:
        dbs = DBSCAN(eps=0.1, min_samples=5).fit(X)
    else:
        dbs = DBSCAN(**best_params['DBSCAN']['params']).fit(X)
    dbs_labels = dbs.labels_
    dbs_time = time.time() - dbs_start

    hdb_start = time.time()
    if 'HDBSCAN' not in best_params:
        hdb = HDBSCAN(min_cluster_size=n_clusters, min_samples=1, cluster_selection_epsilon=0).fit(X)
    else:
        hdb = HDBSCAN(**best_params['HDBSCAN']['params']).fit(X)
    hdb_labels = hdb.labels_
    hdb_time = time.time() - hdb_start

    ms_start = time.time()
    if 'MeanShift' not in best_params:
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=15)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    else:
        params = best_params['MeanShift']['params'].copy()
        bandwidth = estimate_bandwidth(X, quantile=params['quantile'], n_samples=params['n_samples'])
        params['bandwidth'] = bandwidth
        params.pop('quantile', None)
        params.pop('n_samples', None)
        ms = MeanShift(**params).fit(X)
    ms_labels = ms.labels_
    ms_time = time.time() - ms_start

    ac_start = time.time()
    if 'AgglomerativeClustering' not in best_params:
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)
    else:
        ac = AgglomerativeClustering(**best_params['AgglomerativeClustering']['params']).fit(X)
    ac_labels = ac.labels_
    ac_time = time.time() - ac_start

    sc_start = time.time()
    if 'SpectralClustering' not in best_params:
        sc = SpectralClustering(n_clusters=n_clusters, eigen_solver="arpack", affinity="nearest_neighbors", n_neighbors=5, assign_labels='kmeans', random_state=random_state).fit(X)
    else:
        params = best_params['SpectralClustering']['params'].copy()
        if params.get('affinity') == 'rbf' and 'n_neighbors' in params:
            # Remove n_neighbors if affinity is rbf
            params.pop('n_neighbors')
        sc = SpectralClustering(**params).fit(X)
    sc_labels = sc.labels_
    sc_time = time.time() - sc_start

    ed_km_start = time.time()
    ed_Kmeans = ED_KMeansClustering(X, n_clusters, neighbors=5, lookahead=20)
    ed_km_labels, _ = ed_Kmeans.fit(X)
    ed_km_time = time.time() - ed_km_start

    ad_km_start = time.time()
    ad_Kmeans = AD_KMeansClustering(X, n_clusters=n_clusters, n_neighbors=5)
    ad_km_labels = ad_Kmeans.fit(X)
    ad_km_time = time.time() - ad_km_start


    MAP_CLUSTERING_TO_LABELS = {
        "K-Means": (km_labels, km_time),
        "DBSCAN": (dbs_labels, dbs_time),
        "HDBSCAN": (hdb_labels, hdb_time),
        "MeanShift": (ms_labels, ms_time),
        "AgglomerativeClustering": (ac_labels, ac_time),
        "SpectralClustering": (sc_labels, sc_time),
        "ED-K-Means": (ed_km_labels, ed_km_time),
        "AD-K-Means": (ad_km_labels, ad_km_time),
    }

    return MAP_CLUSTERING_TO_LABELS



def run_comparison_clustering_algorithms():
    n_samples = 1000
    datasets = create_set1(n_samples)

    datasets = [
        ("compound", create_compound()),
        ("aggregation", create_aggregation()),
        ("jain", create_jain()),
        ("spiral", create_spiral()),
        ("pathbased", create_pathbased()),
        ("unbalance", create_unbalance()),
    ]
    for data_name, (X, gt) in datasets:
        X = MinMaxScaler(scale).fit_transform(X)

        MAP = run_clustering_algorithms(data_name, X, len(np.unique(gt)))

        print(data_name)

        # prepare a dataframe where rows = algorithms and columns = metrics
        columns = ["ARI", "AMI", "time (s)"]
        df = pd.DataFrame(index=list(MAP.keys()), columns=columns, dtype=float)
        for algo_name, (labels, TIME) in MAP.items():
            ari = adjusted_rand_score(labels, gt)
            ami = adjusted_mutual_info_score(labels, gt)
            df.loc[algo_name, "ARI"] = ari
            df.loc[algo_name, "AMI"] = ami
            df.loc[algo_name, "time (s)"] = TIME

            print(f"{algo_name} in {TIME:.3f}s: {ari:.3f}, {ami:.3f}")

            label_color = [LABEL_COLOR_MAP[i] for i in labels]
            plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
            plt.savefig(FOLDER_FIGS_CLUSTERING + f"/{data_name}_{algo_name}.png")
            plt.savefig(FOLDER_FIGS_CLUSTERING + f"/svgs/{data_name}_{algo_name}.svg")
            # plt.show()
            plt.close()

        # save the dataframe and its transpose
        csv_path = FOLDER_FIGS_CLUSTERING + f"{data_name}.csv"
        df.to_csv(csv_path, float_format="%.3f")
        csv_transpose_path = FOLDER_FIGS_CLUSTERING + f"{data_name}_transpose.csv"
        df.T.to_csv(csv_transpose_path, float_format="%.3f")




def create_paper_tables_from_results():
    import os
    import pandas as pd

    paths = [
        "./results/clustering/aggregation.csv",
        "./results/clustering/compound.csv",
        "./results/clustering/jain.csv",
        "./results/clustering/pathbased.csv",
        "./results/clustering/spiral.csv",
        "./results/clustering/unbalance.csv",
    ]
    per_column = {}
    base_index = None

    for path in paths:
        df = pd.read_csv(path, index_col=0)
        df.index.name = "algorithm"

        if base_index is None:
            base_index = df.index

        file_stem = os.path.splitext(os.path.basename(path))[0]

        for col in df.columns:
            ser = df[col].copy()
            ser.name = file_stem
            per_column.setdefault(col, []).append(ser)

    for col_name, series_list in per_column.items():
        merged = pd.concat(series_list, axis=1, join="outer")
        merged = merged.reindex(base_index)
        merged = merged[sorted(merged.columns)]

        out_fname = f"./paper/tables/clustering_{col_name}.csv"
        merged.to_csv(out_fname, index=True)
        print(f"Saved {out_fname}")




if __name__ == '__main__':
    pass
    run_comparison_clustering_algorithms()
    # create_paper_tables_from_results()
