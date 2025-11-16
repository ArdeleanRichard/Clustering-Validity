import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from load_datasets import create_data3, create_data6, create_data7, create_data4, create_data2, create_data1, create_data5, \
    create_set1
from constants import LABEL_COLOR_MAP

from load_labelsets import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line
from ours.ed_scores import ed_silhouette_score, ed_davies_bouldin_score, ed_calinski_harabasz_score
from ours.ed_kmeans import ED_KMeansClustering
from ours.np_kmeans import KMeansClustering


def run_score_set1(datasets, metric, k, la, plot=False):
    for i_dataset, (X, gt) in enumerate(datasets):
        X, gt = shuffle(X, gt, random_state=7)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        line = diagonal_line(X)
        dp = assign_labels_by_given_line(X, line)

        line = vertical_line(0)
        vl = assign_labels_by_given_line(X, line)

        line = horizontal_line(0)
        hl = assign_labels_by_given_line(X, line)

        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        if metric == 'SS':
            print(f"{silhouette_score(X, gt):.3f}, {ed_silhouette_score(X, gt, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, dp):.3f}, {ed_silhouette_score(X, dp, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, vl):.3f}, {ed_silhouette_score(X, vl, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, hl):.3f}, {ed_silhouette_score(X, hl, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, rl):.3f}, {ed_silhouette_score(X, rl, k=k, lookahead=la):.3f}, \t\t")
        if metric == 'DBS':
            print(f"{davies_bouldin_score(X, gt):.3f}, {ed_davies_bouldin_score(X, gt, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, dp):.3f}, {ed_davies_bouldin_score(X, dp, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, vl):.3f}, {ed_davies_bouldin_score(X, vl, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, hl):.3f}, {ed_davies_bouldin_score(X, hl, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, rl):.3f}, {ed_davies_bouldin_score(X, rl, k=k, lookahead=la, debug=False):.3f}, \t\t")
        if metric == 'CHS':
            print(f"{calinski_harabasz_score(X, gt):.2f}, {ed_calinski_harabasz_score(X, gt, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, dp):.2f}, {ed_calinski_harabasz_score(X, dp, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, vl):.2f}, {ed_calinski_harabasz_score(X, vl, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, hl):.2f}, {ed_calinski_harabasz_score(X, hl, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, rl):.2f}, {ed_calinski_harabasz_score(X, rl, k=k, lookahead=la):.2f}, \t\t")

        if plot:
            for name, labels in zip(["gt", "dp", "vl", "hl", "rl"], [gt, dp, vl, hl, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.svg")
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.png")
                plt.close()

    print()
    # plt.show()


def run_scores_set1(plot=False):
    n_samples = 1000
    datasets = create_set1(n_samples)

    # first run to plot datasets
    run_score_set1(datasets, "", k=0, la=0, plot=plot)

    metric = "SS"
    run_score_set1(datasets, metric, k=3, la=10)
    # run_score_set1(datasets, metric, n_neighbours=15, la=10)
    # run_score_set1(datasets, metric, n_neighbours=5, la=3)

    metric = "DBS"
    run_score_set1(datasets, metric, k=3, la=10)
    # run_score_set1(datasets, metric, n_neighbours=15, la=10)
    # run_score_set1(datasets, metric, n_neighbours=5, la=3)

    metric = "CHS"
    run_score_set1(datasets, metric, k=3, la=10)
    # run_score_set1(datasets, metric, n_neighbours=15, la=10)
    # run_score_set1(datasets, metric, n_neighbours=5, la=3)



def run_score_set2(datasets, metric, k, la, plot=False):
    for i_dataset, (X, gt) in enumerate(datasets):
        X, gt = shuffle(X, gt, random_state=7)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        if metric == 'SS':
            print(f"{silhouette_score(X, gt):.3f}, {ed_silhouette_score(X, gt, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, rl):.3f}, {ed_silhouette_score(X, rl, k=k, lookahead=la):.3f}, \t\t")
        if metric == 'DBS':
            print(f"{davies_bouldin_score(X, gt):.3f}, {ed_davies_bouldin_score(X, gt, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, rl):.3f}, {ed_davies_bouldin_score(X, rl, k=k, lookahead=la, debug=False):.3f}, \t\t")
        if metric == 'CHS':
            print(f"{calinski_harabasz_score(X, gt):.2f}, {ed_calinski_harabasz_score(X, gt, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, rl):.2f}, {ed_calinski_harabasz_score(X, rl, k=k, lookahead=la):.2f}, \t\t")

        if plot:
            for name, labels in zip(["gt", "rl"], [gt, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.svg")
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.png")
                plt.close()

    print()
    # plt.show()



def analyze_time_scores_examples():
    for n_samples in [100, 500, 1000, 5000, 10000]:
        X, gt = create_data3(n_samples)
        ss_start = time.time()
        ss_score = silhouette_score(X, gt)
        ss_time = time.time() - ss_start

        nss_start = time.time()
        nss_score = ed_silhouette_score(X, gt, k=5, lookahead=10, debug=False)
        nss_time = time.time() - nss_start
        print(f"D3 with {n_samples}samples, SS: {ss_score:.3f} in {ss_time:.3f}s, NSS: {nss_score:.3f} in {nss_time:.3f}s")
        print()



def analyze_time_scores_features():
    n_samples = 1000
    for n_features in [2,3,4,5,6]:
        X, gt = create_data3(n_samples, n_features)
        ss_start = time.time()
        ss_score = silhouette_score(X, gt)
        ss_time = time.time() - ss_start


        nss_start = time.time()
        nss_score = ed_silhouette_score(X, gt, debug=True)
        nss_time = time.time() - nss_start
        print(f"D3 with {n_features}features, SS: {ss_score:.3f} in {ss_time:.3f}s, NSS: {nss_score:.3f} in {nss_time:.3f}s")
        print()



def run_kmeans_set1():
    n_samples = 1000
    datasets = create_set1(n_samples)

    for i_dataset, (X, gt) in enumerate(datasets):
        label_color = [LABEL_COLOR_MAP[i] for i in gt]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.png")
        plt.close()

        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)

        km = KMeans(n_clusters=2, ).fit(X)
        k_labels = km.labels_

        label_color = [LABEL_COLOR_MAP[i] for i in k_labels]
        # plt.title(f"K-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.png")
        # plt.show()
        plt.close()


        ed_Kmeans = ED_KMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = ed_Kmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in nk_labels]
        # plt.title(f"NewK-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.png")
        # plt.show()
        plt.close()



        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_

        label_color = [LABEL_COLOR_MAP[i] for i in s_labels]
        # plt.title(f"NewK-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_sc.png")
        # plt.show()
        plt.close()

        print(f"{adjusted_rand_score(k_labels, gt):.3f}, {adjusted_rand_score(nk_labels, gt):.3f}, {adjusted_rand_score(s_labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(k_labels, gt):.3f}, {adjusted_mutual_info_score(nk_labels, gt):.3f}, {adjusted_mutual_info_score(s_labels, gt):.3f}")
        print()




def analysis_kmeans_by_metrics():
    n_samples = 1000

    data1 = create_data1(n_samples)
    data2 = create_data2(n_samples)
    data3 = create_data3(n_samples)
    data4 = create_data4(n_samples)
    data5 = create_data5(n_samples)
    data6 = create_data6(n_samples)
    data7 = create_data7(n_samples)

    datasets = [
        data1,
        data2,
        data3,
        data4,
        data5,
        data6,
        data7,
    ]

    for i_dataset, (X, gt) in enumerate(datasets):
        label_color = [LABEL_COLOR_MAP[i] for i in gt]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.png")
        plt.close()

        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in k_labels]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.png")
        plt.show()
        plt.close()

        ed_Kmeans = ED_KMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = ed_Kmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in nk_labels]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.png")
        plt.show()
        plt.close()

        print(f"{adjusted_rand_score(k_labels, gt):.3f}, {adjusted_rand_score(nk_labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(k_labels, gt):.3f}, {adjusted_mutual_info_score(nk_labels, gt):.3f}")
        print()





def analyze_time_kmeans_examples():
    for n_samples in [100, 500, 1000, 5000, 10000]:
        X, gt = create_data3(n_samples)

        k_start = time.time()
        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)
        k_time = time.time() - k_start

        nk_start = time.time()
        ed_Kmeans = ED_KMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = ed_Kmeans.fit(X)
        nk_time = time.time() - nk_start

        s_start = time.time()
        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_
        s_time = time.time() - s_start

        print(f"D3 with {n_samples}samples, "
              f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
              f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s"
              f"SpectralClustering ({adjusted_rand_score(s_labels, gt):.3f}/{adjusted_mutual_info_score(s_labels, gt):.3f}): {s_time:.3f}s"
              )




def analyze_time_kmeans_features():
    for n_features in [2,3,4,5,6]:
        n_samples = 1000
        X, gt = create_data3(n_samples, n_features)

        k_start = time.time()
        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)
        k_time = time.time() - k_start

        nk_start = time.time()
        ed_Kmeans = ED_KMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = ed_Kmeans.fit(X)
        nk_time = time.time() - nk_start

        s_start = time.time()
        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_
        s_time = time.time() - s_start

        print(f"D3 with {n_samples}samples, "
              f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
              f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s"
              f"SpectralClustering ({adjusted_rand_score(s_labels, gt):.3f}/{adjusted_mutual_info_score(s_labels, gt):.3f}): {s_time:.3f}s"
              )



if __name__ == '__main__':
    pass
    # run_scores_set1()

    # analyze_time_scores_examples()
    # analyze_time_scores_features()

    # run_kmeans_set1()

    # analyze_time_kmeans_examples()
    # analyze_time_kmeans_features()