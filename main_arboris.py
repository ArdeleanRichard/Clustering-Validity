from cvis_ours.ad_CVIs import ad_silhouette_score, ad_davies_bouldin_score, ad_calinski_harabasz_score, arboris_index

def main_compare_performance():
    from time import time
    from load_datasets import create_data4
    from sklearn.preprocessing import MinMaxScaler

    for n in [500, 1000, 2000, 5000, 10000, 15000, 20000]:
        print(f"\nDataset size: {n} samples")
        X, labels = create_data4(n)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        k = 5
        # Silhouette score
        start = time()
        score = ad_silhouette_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Silhouette: {score:.4f} in {timee:.3f}s")

        # Davies-Bouldin score
        start = time()
        score = ad_davies_bouldin_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Davies-Bouldin: {score:.4f} in {timee:.3f}s")

        # Calinski-Harabasz score
        start = time()
        score = ad_calinski_harabasz_score(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  AD Calinski-Harabasz: {score:.4f} in {timee:.3f}s")

        # Calinski-Harabasz score
        start = time()
        score = arboris_index(X, labels, n_neighbors=k)
        timee = time() - start
        print(f"  Arboris: {score:.4f} in {timee:.3f}s")

# Example usage
if __name__ == "__main__":
     main_compare_performance()
