import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from constants import FOLDER_FIGS_ANALYSIS_ESTIMATE, random_state
from constants_maps import MAP_INTERNAL_METRICS
from load_datasets import create_data4


def compute_centroids(X, labels):
    unique_labels = np.unique(labels)
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
    return centroids


def scatter_plot(ax, title, cvi_str, X, nr_clusts, labels=None):
    cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]
    # For visualization, use first 2 dimensions if data is high-dimensional
    if X.shape[1] == 2:
        X_plot = X
        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    else:
        # Use PCA for visualization if more than 2 features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=random_state)
        X_plot = pca.fit_transform(X)
        ax.set_xlabel('First Principal Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Second Principal Component', fontsize=11, fontweight='bold')

    if labels is None:
        kmeans = KMeans(n_clusters=nr_clusts, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

    scatter2 = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Plot centroids
    if labels is None:
        if X.shape[1] == 2:
            centroids = kmeans.cluster_centers_
        else:
            centroids = pca.transform(kmeans.cluster_centers_)
    else:
        centroids = compute_centroids(X, labels)

    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids', zorder=5)

    k_score = cvi_function(X, labels)
    ax.set_title(title + f"\n({cvi_name_full}: {k_score:.4f})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.colorbar(scatter2, ax=ax, label='Cluster')



def analyze_score(cvi_str, X, true_labels, k_range=None, dataset_name="dataset"):
    """
    Analyze K-Means clustering using cvi to estimate optimal K.

    Parameters:
    -----------
    X : np.ndarray
        Input data of shape (n_samples, n_features)
    true_k : int
        The true/correct number of clusters
    k_range : tuple, optional
        Range of K values to test as (min_k, max_k).
        Default is (2, min(10, n_samples-1))
    dataset_name : str
        Name of the dataset for labeling outputs
    output_dir : str
        Directory to save results
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing K values and their scores
    """
    cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]

    true_k = len(np.unique(true_labels))

    # Set default k_range if not provided
    if k_range is None:
        k_range = (2, 10)

    min_k, max_k = k_range

    # Validate inputs
    if min_k < 2:
        raise ValueError("Minimum K must be at least 2")
    if max_k > X.shape[0] - 1:
        raise ValueError(f"Maximum K must be less than n_samples ({X.shape[0]})")
    if true_k < 2:
        raise ValueError("True K must be at least 2")

    # Calculate scores for different K values
    k_values = range(min_k, max_k + 1)
    scores = []

    print(f"Analyzing {dataset_name}...")
    print(f"Testing K values from {min_k} to {max_k}")
    print(f"True K: {true_k}")
    print("-" * 50)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = cvi_function(X, labels)
        scores.append(score)
        print(f"K={k}: {cvi_name_full} = {score:.4f}")

    # Find the K with maximum score
    optimal_idx = np.argmax(scores)
    estimated_k = k_values[optimal_idx]

    print("-" * 50)
    print(f"Estimated K (max {cvi_name_full}): {estimated_k}")
    print(f"True K: {true_k}")
    print(f"Match: {'✓ YES' if estimated_k == true_k else '✗ NO'}")
    print(f"Max {cvi_name_full}: {scores[optimal_idx]:.4f}")

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'K': list(k_values),
        f'{cvi_name_full}': scores,
        'Is_Estimated_K': [k == estimated_k for k in k_values],
        'Is_True_K': [k == true_k for k in k_values]
    })

    # Save results to CSV
    csv_path = f"{FOLDER_FIGS_ANALYSIS_ESTIMATE}/{dataset_name}_{cvi_str}_scores.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))

    # Plot 1: scores
    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(k_values, scores, 'b-o', linewidth=2, markersize=6, label=f'{cvi_name_full}')

    # Mark the estimated K (max score)
    ax1.plot(estimated_k, scores[optimal_idx], 'r*', markersize=20, label=f'Estimated K={estimated_k}', zorder=5)

    # Mark the true K
    if true_k in k_values:
        true_k_idx = list(k_values).index(true_k)
        ax1.plot(true_k, scores[true_k_idx], 'g^', markersize=15, label=f'True K={true_k}', zorder=5)
    else:
        # If true_k is outside the range, add a vertical line
        ax1.axvline(x=true_k, color='g', linestyle='--', linewidth=2, label=f'True K={true_k} (outside range)')

    ax1.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{cvi_name_full}', fontsize=11, fontweight='bold')
    ax1.set_title(f'{cvi_name_full} vs K', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xticks(k_values)

    # Add annotation for match/mismatch
    # match_text = "✓ MATCH" if estimated_k == true_k else "✗ MISMATCH"
    # match_color = "green" if estimated_k == true_k else "red"
    # ax1.text(0.02, 0.98, match_text, transform=ax1.transAxes,
    #          fontsize=11, fontweight='bold', color=match_color,
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))



    # Plot 2: K-Means clustering with estimated K
    ax2 = plt.subplot(1, 4, 2)
    scatter_plot(ax2, f'K-Means with estimated K={estimated_k}', cvi_str, X, estimated_k)

    # Plot 3: K-Means clustering with true K
    ax3 = plt.subplot(1, 4, 3)
    scatter_plot(ax3, f'K-Means with true K={true_k}', cvi_str, X, true_k)

    # Plot 4: True labels
    ax4 = plt.subplot(1, 4, 4)
    scatter_plot(ax4, f'True labels', cvi_str, X, true_k, labels=true_labels)


    # Main title
    fig.suptitle(f'K-Means {cvi_name_full} Analysis - {dataset_name}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save plot
    plot_path = f"{FOLDER_FIGS_ANALYSIS_ESTIMATE}/{dataset_name}_{cvi_str}_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.close()

    return results_df


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate sample data with 4 clusters
    # X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=0.6, random_state=42)

    n_samples = 1000
    X, y_true = create_data4(n_samples)

    # Analyze with Silhouette score
    results = analyze_score(
        cvi_str="silhouette",
        X=X,
        true_labels=y_true,
        k_range=(2, 10),
        dataset_name="data4"
    )

    print("\n" + "=" * 50)
    print("Results DataFrame:")
    print(results)