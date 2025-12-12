import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, estimate_bandwidth, MeanShift, AgglomerativeClustering, DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from load_datasets import create_data3, create_data6, create_data7, create_data4, create_data2, create_data1, create_data5, create_set1, create_compound, create_aggregation, create_jain, create_unbalance, create_spiral, create_pathbased
from constants import LABEL_COLOR_MAP, random_state, scale
from ours.np_kmeans import KMeansClustering
from ours.ed_kmeans import ED_KMeansClustering
from ours.ad_kmeans import AD_KMeansClustering




def run_clustering_algorithms(X, n_clusters):
    km_start = time.time()
    # Kmeans = KMeansClustering(X, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    km_labels = kmeans.labels_
    km_time = time.time() - km_start

    dbs_start = time.time()
    dbs = DBSCAN(eps=0.35, min_samples=int(np.log(len(X)))).fit(X)
    dbs_labels = dbs.labels_
    dbs_time = time.time() - dbs_start

    hdb_start = time.time()
    hdb = HDBSCAN(min_cluster_size=5).fit(X)
    hdb_labels = hdb.labels_
    hdb_time = time.time() - hdb_start

    ms_start = time.time()
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=50)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    ms_labels = ms.labels_
    ms_time = time.time() - ms_start

    ac_start = time.time()
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)
    ac_labels = ward.labels_
    ac_time = time.time() - ac_start

    sc_start = time.time()
    sc = SpectralClustering(n_clusters=n_clusters, eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
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



def run_kmeans_set1():
    n_samples = 1000
    datasets = create_set1(n_samples)

    datasets = [
        # ("compound", create_compound()),
        # ("aggregation", create_aggregation()),
        # ("jain", create_jain()),
        ("spiral", create_spiral()),
        ("pathbased", create_pathbased()),
        ("unbalance", create_unbalance()),
    ]
    for data_name, (X, gt) in datasets:
        X = MinMaxScaler(scale).fit_transform(X)

        MAP = run_clustering_algorithms(X, len(np.unique(gt)))

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

            print(f"{algo_name} in {TIME}s: {ari:.3f}, {ami:.3f}")

            label_color = [LABEL_COLOR_MAP[i] for i in labels]
            plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
            plt.savefig(f"./figs/kmeans/{data_name}_{algo_name}.png")
            plt.savefig(f"./figs/kmeans/svgs/{data_name}_{algo_name}.svg")
            # plt.show()
            plt.close()

        # save the dataframe and its transpose
        csv_path = f"./results/kmeans/{data_name}.csv"
        df.to_csv(csv_path, float_format="%.3f")
        csv_transpose_path = f"./results/kmeans/{data_name}_transpose.csv"
        df.T.to_csv(csv_transpose_path, float_format="%.3f")




# def analyze_time_kmeans_examples():
#     MAP_CLUSTERING_TO_LABELS = {}
#     for n_samples in [100, 500, 1000, 5000, 10000]:
#         X, gt = create_data3(n_samples)
#
#
#
#         print(f"D3 with {n_samples}samples, "
#               f"KMeans ({adjusted_rand_score(km_labels, gt):.3f}/{adjusted_mutual_info_score(km_labels, gt):.3f}): {km_time:.3f}s, "
#               f"SpectralClustering ({adjusted_rand_score(sc_labels, gt):.3f}/{adjusted_mutual_info_score(sc_labels, gt):.3f}): {sc_time:.3f}s"
#               f"ED-KMeans ({adjusted_rand_score(ed_km_labels, gt):.3f}/{adjusted_mutual_info_score(ed_km_labels, gt):.3f}): {ed_km_time:.3f}s"
#               f"AD-KMeans ({adjusted_rand_score(ad_km_labels, gt):.3f}/{adjusted_mutual_info_score(ad_km_labels, gt):.3f}): {ad_km_time:.3f}s"
#               )
#
#
# def analyze_time_kmeans_features():
#     for n_features in [2,3,4,5,6]:
#         n_samples = 1000
#         X, gt = create_data3(n_samples, n_features)
#
#         k_start = time.time()
#         Kmeans = KMeansClustering(X, len(np.unique(gt)))
#         k_labels = Kmeans.fit(X)
#         k_time = time.time() - k_start
#
#         nk_start = time.time()
#         ed_Kmeans = ED_KMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
#         nk_labels, centroids = ed_Kmeans.fit(X)
#         nk_time = time.time() - nk_start
#
#         s_start = time.time()
#         sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
#         s_labels = sc.labels_
#         s_time = time.time() - s_start
#
#         print(f"D3 with {n_samples}samples, "
#               f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
#               f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s"
#               f"SpectralClustering ({adjusted_rand_score(s_labels, gt):.3f}/{adjusted_mutual_info_score(s_labels, gt):.3f}): {s_time:.3f}s"
#               )



if __name__ == '__main__':
    pass
    run_kmeans_set1()

    # analyze_time_kmeans_examples()
    # analyze_time_kmeans_features()