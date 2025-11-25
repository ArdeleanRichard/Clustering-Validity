import numpy as np
from matplotlib import pyplot as plt

from constants import random_state, LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS
from load_datasets import generate_clusters_analysis
from ours.measures import imbalance_ratio, overlap_ratio


# -------------------- Plot helpers --------------------

def save_fig(fig, filename, dpi=300):
    fig.savefig(f"./{FOLDER_FIGS_ANALYSIS}/{filename}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: ./{FOLDER_FIGS_ANALYSIS}/{filename}")


def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


# -------------------- Analysis: Overlap vs centroid distance --------------------

def analyze_cluster_overlap(
    distances=np.linspace(6, 1, 20),
    n_per_cluster=300,
    cluster_std=1.0,
    save_prefix="analysis_overlap",
):
    """
    Generate three clusters where cluster 0 and 1 move closer by `distances`.
    Compute R-value for each distance and plot results + example scatter plots.
    Returns (centroid_distances, overlap_ratios).
    """
    set_plot_style()

    overlap_ratios = []
    centroid_distances = []
    small_examples = []  # store small datasets for plotting examples

    for d in distances:
        centers = [(0.0, 0.0), (d, 0.0), (3.0, 5.0)]
        sizes = [n_per_cluster, n_per_cluster, n_per_cluster]
        # generate a larger dataset for metric computation
        X, labels = generate_clusters_analysis(centers, sizes, cluster_std=cluster_std)
        r_val = overlap_ratio(X, labels)
        overlap_ratios.append(r_val)
        centroid_distances.append(float(np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))))

        # generate a smaller dataset for plotting examples (keeps variety with rng)
        X_small, labels_small = generate_clusters_analysis(centers, [100, 100, 100], cluster_std=cluster_std)
        small_examples.append((X_small, labels_small, centers))

    # Plot Distance vs Overlap
    fig1, ax1 = plt.subplots()
    ax1.plot(centroid_distances, overlap_ratios, 'o-', linewidth=3, markersize=8)
    ax1.set_xlabel('Distance Between Centroids (Clusters 0 & 1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overlap Ratio (OR)', fontsize=14, fontweight='bold')
    ax1.set_title('Centroid Distance vs. Cluster Overlap', fontsize=16, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.tick_params(labelsize=11)

    # Annotations
    # ax1.annotate('Far - Low Overlap',
    #              xy=(centroid_distances[0], overlap_ratios[0]),
    #              xytext=(centroid_distances[0] - 0.6, overlap_ratios[0] + 0.06),
    #              fontsize=10, ha='right',
    #              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # ax1.annotate('Close - High Overlap',
    #              xy=(centroid_distances[-1], overlap_ratios[-1]),
    #              xytext=(centroid_distances[-1] + 0.6, overlap_ratios[-1] - 0.06),
    #              fontsize=10, ha='left',
    #              bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))

    save_fig(fig1, f"{save_prefix}.png")

    # Example scatter plots: pick far, mid, close
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    example_indices = [0, len(distances) // 2, -1]
    for ax, idx in zip(axes2, example_indices):
        Xs, labs, centers = small_examples[idx]
        for k in range(3):
            mask = labs == k
            ax.scatter(Xs[mask, 0], Xs[mask, 1], c=LABEL_COLOR_MAP[k], alpha=0.6, s=30, label=f"Cluster {k}")
            ax.scatter(*centers[k], c=LABEL_COLOR_MAP[k], marker='X', s=200, edgecolors='black', linewidths=2, zorder=5)
        d_val = distances[idx]
        ax.set_title(f"distance={d_val:.1f}, OR={overlap_ratios[idx]:.2f}", fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 8)
        ax.set_ylim(-3, 8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig2, f"{save_prefix}_examples.png")

    return centroid_distances, overlap_ratios



# -------------------- Analysis: Imbalance Ratio --------------------

def analyze_imbalance_ratio(
    initial_size=300,
    cluster_std=1.0,
    cluster_2_sizes=None,
    centers=None,
    save_prefix="analysis_imbalance"
):
    """
    Vary the size of the minority cluster (cluster index 2) and compute Imbalance Ratio (IR).
    Returns (minority_sizes, imbalance_ratios) and also saves:
      - a line plot of IR vs minority cluster size
      - a 3-panel figure showing example scatter plots (start / middle / end)
    """
    set_plot_style()
    if cluster_2_sizes is None:
        cluster_2_sizes = np.linspace(initial_size, int(initial_size * 0.1), 20).astype(int)
    if centers is None:
        centers = [(0, 0), (5, 0), (2.5, 4)]

    imbalance_ratios = []
    minority_sizes = []
    small_examples = []  # store small datasets for plotting examples

    for size_2 in cluster_2_sizes:
        # full dataset for metric computation
        X, labels = generate_clusters_analysis(centers, [initial_size, initial_size, int(size_2)], cluster_std=cluster_std)
        ir = imbalance_ratio(None, labels)
        imbalance_ratios.append(ir)
        minority_sizes.append(int(size_2))

        # small dataset for plotting (keeps same relative sizes)
        # use smaller per-cluster counts for visualization clarity
        small_sizes = [min(100, initial_size), min(100, initial_size), min(100, int(size_2))]
        X_small, labels_small = generate_clusters_analysis(centers, small_sizes, cluster_std=cluster_std)
        small_examples.append((X_small, labels_small, centers))

    # --- Plot IR vs minority size ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(minority_sizes, imbalance_ratios, 'o-', linewidth=2, markersize=7)
    ax.set_xlabel('Minority Cluster Size (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imbalance Ratio (IR)', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Size vs. Imbalance Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.axhline(y=1.0, linestyle='--', linewidth=2, label='Balanced (IR=1)')
    ax.legend()

    save_fig(fig, f"{save_prefix}.png")

    # --- Example scatter plots: pick start, middle, end ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    example_indices = [0, len(minority_sizes) // 2, -1]

    for ax, idx in zip(axes2, example_indices):
        Xs, labs, ctrs = small_examples[idx]
        for k in range(3):
            mask = labs == k
            ax.scatter(Xs[mask, 0], Xs[mask, 1], c=LABEL_COLOR_MAP[k], alpha=0.6, s=30, label=f"Cluster {k}")
            # center marker
            ax.scatter(*ctrs[k], c=LABEL_COLOR_MAP[k], marker='X', s=200,
                       edgecolors='black', linewidths=2, zorder=5)
        size_val = minority_sizes[idx]
        ir_val = imbalance_ratios[idx]
        ax.set_title(f'n_minority={size_val}, IR={ir_val:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 8)
        ax.set_ylim(-3, 8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig2, f"{save_prefix}_examples.png")

    return minority_sizes, imbalance_ratios




def safe_percent_change(start, end):
    """Return percent change from start to end. If start == 0, return None."""
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0






if __name__ == "__main__":
    print("=== IMBALANCE MEASURE ===")
    # Example: Binary classification with imbalance
    y_binary = np.array([0] * 100 + [1] * 900)
    ir = imbalance_ratio(None, y_binary)
    print(f"Imbalance Ratio: {ir:.2f}")
    print(f"Class Distribution: {np.unique(y_binary, return_counts=True)}")
    print()

    print("=== CLUSTER OVERLAP MEASURE ===")
    # Example 3: Clustering with overlap
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.5, random_state=random_state)
    # R-value
    r_val = overlap_ratio(X, y_true)
    print(f"OR-value (Overlap Rate): {r_val:.3f} ({r_val * 100:.1f}% overlapping)")


    print("=" * 60)
    print("ANALYSIS 1: CLUSTER OVERLAP VS CENTROID DISTANCE")
    print("=" * 60)
    centroid_dists, overlap_ratios = analyze_cluster_overlap(
        distances=np.linspace(6, 1, 20),
        n_per_cluster=300,
        cluster_std=1.0,
        save_prefix="overlap"
    )
    print(f"Distance range: {centroid_dists[0]:.2f} to {centroid_dists[-1]:.2f}")
    print(f"Overlap ratio range: {overlap_ratios[0]:.3f} to {overlap_ratios[-1]:.3f}")
    pct = safe_percent_change(overlap_ratios[0], overlap_ratios[-1])
    if pct is None:
        print("Cannot compute percent change for overlap (initial value is 0).")
    else:
        print(f"As centroids get closer, overlap increases by {pct:.1f}%")


    print("\n" + "=" * 60)
    print("ANALYSIS 2: IMBALANCE RATIO VS CLUSTER SIZE")
    print("=" * 60)
    minority_sizes, imb_ratios = analyze_imbalance_ratio(
        initial_size=300,
        cluster_std=1.0,
        cluster_2_sizes=np.linspace(300, 30, 20).astype(int),
        centers=[(0, 0), (5, 0), (2.5, 4)],
        save_prefix="imbalance",
    )
    print(f"Minority cluster size range: {minority_sizes[0]} to {minority_sizes[-1]}")
    print(f"Imbalance ratio range: {imb_ratios[0]:.2f} to {imb_ratios[-1]:.2f}")
    pct2 = safe_percent_change(imb_ratios[0], imb_ratios[-1])
    if pct2 is None:
        print("Cannot compute percent change for imbalance (initial value is 0).")
    else:
        print(f"As minority cluster shrinks, imbalance increases by {pct2:.1f}%")