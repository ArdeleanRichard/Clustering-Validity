"""
Refactored cluster analysis:
- Vectorized overlap (R-value) computation
- Reusable helpers for generating clusters, plotting and saving figures
- Parameters exposed for easier experimentation
- More concise, readable main flow
"""
from typing import List, Tuple, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from constants import LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS

# -------------------- Utilities & Metrics --------------------

def generate_clusters(
    centers: Sequence[Sequence[float]],
    sizes: Sequence[int],
    cluster_std: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset from Gaussian blobs centered at `centers` with sample counts `sizes`.
    Returns (X, labels) where X is (N, 2) and labels is length N.
    """
    if rng is None:
        rng = np.random.default_rng()
    centers = np.asarray(centers)
    X_parts = []
    labels_parts = []
    for i, (c, n) in enumerate(zip(centers, sizes)):
        X_i = rng.normal(loc=0.0, scale=cluster_std, size=(n, centers.shape[1])) + np.asarray(c)
        X_parts.append(X_i)
        labels_parts.append(np.full(n, i, dtype=int))
    X = np.vstack(X_parts)
    labels = np.concatenate(labels_parts)
    return X, labels


def imbalance_ratio(labels: Sequence[int]) -> float:
    """
    Imbalance Ratio (IR) = majority_class_size / minority_class_size.
    If a class has 0 members, returns np.inf.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)

    max_count = np.max(counts)
    min_count = np.max(counts)
    if min_count == 0:
        return np.inf

    return max_count / min_count


def overlap_rate_rvalue(X: np.ndarray, labels: np.ndarray, slack: float = 1.2) -> float:
    """
    Vectorized computation of the R-value (overlap rate).
    For each point, compute distance to its cluster center and the nearest other center.
    Point is 'overlapping' if dist_to_nearest_other <= slack * dist_to_own.
    Returns fraction of overlapping points.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array (N, dims)")
    unique_labels, inv = np.unique(labels, return_inverse=True)
    n_centers = unique_labels.size
    dims = X.shape[1]

    # compute centers in the order of unique_labels
    centers = np.vstack([X[labels == lab].mean(axis=0) for lab in unique_labels])

    # distances: shape (N_points, n_centers)
    # broadcasting: points[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    # own center distance for each point
    own_center_idx = inv  # maps each point to index in centers
    idx = np.arange(X.shape[0])
    dist_to_own = dists[idx, own_center_idx]

    # distance to nearest other center: set own center distance to +inf and take min
    dists_other = dists.copy()
    dists_other[idx, own_center_idx] = np.inf
    dist_to_nearest_other = dists_other.min(axis=1)

    # overlapping condition
    overlapping = dist_to_nearest_other <= (slack * dist_to_own)
    overlap_rate = overlapping.sum() / X.shape[0]
    return overlap_rate


def safe_percent_change(start: float, end: float) -> Optional[float]:
    """Return percent change from start to end. If start == 0, return None."""
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0


# -------------------- Plot helpers --------------------

def save_fig(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
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


# -------------------- Analysis 1: Overlap vs centroid distance --------------------

def analyze_cluster_overlap(
    distances: Sequence[float] = np.linspace(6, 1, 20),
    n_per_cluster: int = 300,
    cluster_std: float = 1.0,
    save_prefix: str = "analysis_overlap",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[float], List[float]]:
    """
    Generate three clusters where cluster 0 and 1 move closer by `distances`.
    Compute R-value for each distance and plot results + example scatter plots.
    Returns (centroid_distances, overlap_ratios).
    """
    set_plot_style()
    if rng is None:
        rng = np.random.default_rng(42)

    overlap_ratios = []
    centroid_distances = []
    small_examples = []  # store small datasets for plotting examples

    for d in distances:
        centers = [(0.0, 0.0), (d, 0.0), (3.0, 5.0)]
        sizes = [n_per_cluster, n_per_cluster, n_per_cluster]
        # generate a larger dataset for metric computation
        X, labels = generate_clusters(centers, sizes, cluster_std=cluster_std, rng=rng)
        r_val = overlap_rate_rvalue(X, labels)
        overlap_ratios.append(r_val)
        centroid_distances.append(float(np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))))

        # generate a smaller dataset for plotting examples (keeps variety with rng)
        X_small, labels_small = generate_clusters(centers, [100, 100, 100], cluster_std=cluster_std, rng=rng)
        small_examples.append((X_small, labels_small, centers))

    # Plot Distance vs Overlap
    fig1, ax1 = plt.subplots()
    ax1.plot(centroid_distances, overlap_ratios, 'o-', linewidth=3, markersize=8)
    ax1.set_xlabel('Distance Between Centroids (Clusters 0 & 1)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Overlap Ratio (R-value)', fontsize=14, fontweight='bold')
    ax1.set_title('Centroid Distance vs. Cluster Overlap', fontsize=16, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    ax1.tick_params(labelsize=11)

    # Annotations
    ax1.annotate('Far - Low Overlap',
                 xy=(centroid_distances[0], overlap_ratios[0]),
                 xytext=(centroid_distances[0] - 0.6, overlap_ratios[0] + 0.06),
                 fontsize=10, ha='right',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
    ax1.annotate('Close - High Overlap',
                 xy=(centroid_distances[-1], overlap_ratios[-1]),
                 xytext=(centroid_distances[-1] + 0.6, overlap_ratios[-1] - 0.06),
                 fontsize=10, ha='left',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

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
        ax.set_title(f"d={d_val:.1f}, R={overlap_ratios[idx]:.2f}", fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 8)
        ax.set_ylim(-3, 8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig2, f"{save_prefix}_examples.png")

    return centroid_distances, overlap_ratios


# -------------------- Analysis 2: Imbalance Ratio --------------------

def analyze_imbalance_ratio(
    initial_size: int = 300,
    cluster_std: float = 1.0,
    cluster_2_sizes: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    save_prefix: str = "analysis_imbalance",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], List[float]]:
    """
    Vary the size of the minority cluster (cluster index 2) and compute Imbalance Ratio (IR).
    Returns (minority_sizes, imbalance_ratios) and also saves:
      - a line plot of IR vs minority cluster size
      - a 3-panel figure showing example scatter plots (start / middle / end)
    """
    set_plot_style()
    if rng is None:
        rng = np.random.default_rng(42)
    if cluster_2_sizes is None:
        cluster_2_sizes = np.linspace(initial_size, int(initial_size * 0.1), 20).astype(int)
    if centers is None:
        centers = [(0, 0), (5, 0), (2.5, 4)]

    imbalance_ratios = []
    minority_sizes = []
    small_examples = []  # store small datasets for plotting examples

    for size_2 in cluster_2_sizes:
        # full dataset for metric computation
        X, labels = generate_clusters(centers, [initial_size, initial_size, int(size_2)],
                                      cluster_std=cluster_std, rng=rng)
        ir = imbalance_ratio(labels)
        imbalance_ratios.append(ir)
        minority_sizes.append(int(size_2))

        # small dataset for plotting (keeps same relative sizes)
        # use smaller per-cluster counts for visualization clarity
        small_sizes = [min(100, initial_size), min(100, initial_size), min(100, int(size_2))]
        X_small, labels_small = generate_clusters(centers, small_sizes, cluster_std=cluster_std, rng=rng)
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


# -------------------- Analysis 3: Overlap vs Silhouette --------------------

def analyze_overlap_silhouette(
    distances: Sequence[float] = np.linspace(6, 1, 20),
    n_per_cluster: int = 300,
    cluster_std: float = 1.0,
    save_filename: str = "analysis_overlap_silhouette.png",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For varying centroid distances, compute overlap (R), silhouette with true labels,
    silhouette with a horizontal split at y_mid. Plot silhouette vs overlap and an example scatter.
    """
    set_plot_style()
    if rng is None:
        rng = np.random.default_rng(42)

    overlap_list = []
    ss_tl = []
    ss_hl = []
    datasets = []

    for d in distances:
        centers = [(0.0, 0.0), (d, 0.0), (3.0, 5.0)]
        X, labels_true = generate_clusters(centers, [n_per_cluster, n_per_cluster, n_per_cluster], cluster_std=cluster_std, rng=rng)

        r = overlap_rate_rvalue(X, labels_true)
        overlap_list.append(r)

        # silhouette for true labels
        s_true = silhouette_score(X, labels_true)
        ss_tl.append(s_true)

        # horizontal split at midpoint of Y
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        y_mid = (y_min + y_max) / 2.0
        labels_h = (X[:, 1] > y_mid).astype(int)

        s_h = silhouette_score(X, labels_h)
        ss_hl.append(s_h)

        datasets.append((X, labels_h, labels_true, y_mid, d))

    overlap_arr = np.asarray(overlap_list)
    ss_tl_arr = np.asarray(ss_tl)
    ss_hl_arr = np.asarray(ss_hl)

    # pick example where horizontal > true, otherwise pick the max horizontal
    better_idx = np.where(ss_hl_arr > ss_tl_arr)[0]
    chosen_idx = int(better_idx[0]) if better_idx.size > 0 else int(np.nanargmax(ss_hl_arr))

    X_plot, horiz_labels, true_labels, y_mid, dist_val = datasets[chosen_idx]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    axes[0].plot(overlap_arr, ss_tl_arr, marker='o', linewidth=2, label='Silhouette (TL)')
    axes[0].plot(overlap_arr, ss_hl_arr, marker='s', linewidth=2, label='Silhouette (HL)')
    axes[0].axvline(x=overlap_arr[chosen_idx], color='gray', linestyle='--', alpha=0.6)
    axes[0].text(overlap_arr[chosen_idx], np.nanmax([np.nanmax(ss_tl_arr), np.nanmax(ss_hl_arr)]) * 0.9,
                 f"example shown\nR={overlap_arr[chosen_idx]:.2f}",
                 ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axes[0].set_xlabel('Overlap Rate (R-value)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Overlap vs Silhouette', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    label_color = [LABEL_COLOR_MAP[i] for i in horiz_labels]
    axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=label_color, s=25, alpha=0.7)
    axes[1].axhline(y=y_mid, color='black', linestyle='--', linewidth=2, alpha=0.8)
    axes[1].set_title(
        f'Horizontal split labels\nDistance={dist_val:.2f}, R={overlap_arr[chosen_idx]:.2f}\n'
        f'SS(HL)={ss_hl_arr[chosen_idx]:.3f} > SS(TL)={ss_tl_arr[chosen_idx]:.3f}',
        fontsize=11, fontweight='bold'
    )
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(alpha=0.3)

    label_color = [LABEL_COLOR_MAP[i] for i in true_labels]
    axes[2].scatter(X_plot[:, 0], X_plot[:, 1], c=label_color, s=25, alpha=0.7)
    axes[2].set_title(
        f'True labels \nDistance={dist_val:.2f}, R={overlap_arr[chosen_idx]:.2f}\n',
        fontsize=11, fontweight='bold'
    )
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_filename)

    return overlap_arr, ss_tl_arr, ss_hl_arr



def analyze_imbalance_silhouette(
    initial_size: int = 300,
    cluster_std: float = 1.0,
    cluster_2_sizes: Optional[Sequence[int]] = None,
    centers: Optional[Sequence[Sequence[float]]] = None,
    save_filename: str = "analysis_imbalance_silhouette.png",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vary the minority cluster size and compute:
      - Imbalance Ratio (IR)
      - Silhouette for true labels (3 clusters)
      - Silhouette for horizontal midpoint split (2 labels)
    Plot Silhouette vs IR and display an example where the horizontal split outperforms
    the true labeling (or the horizontal split with max silhouette if none).
    Returns (imbalance_arr, ss_tl_arr, ss_hl_arr).
    """
    set_plot_style()
    if rng is None:
        rng = np.random.default_rng(42)
    if cluster_2_sizes is None:
        cluster_2_sizes = np.linspace(initial_size, int(initial_size * 0.1), 20).astype(int)
    if centers is None:
        centers = [(0, 0), (5, 0), (2.5, 4)]

    imbalance_list = []
    ss_tl_list = []
    ss_hl_list = []
    datasets = []

    for size_2 in cluster_2_sizes:
        X, labels_true = generate_clusters(centers, [initial_size, initial_size, int(size_2)],
                                           cluster_std=cluster_std, rng=rng)

        # Imbalance ratio
        ir = imbalance_ratio(labels_true)
        imbalance_list.append(ir)

        # silhouette for true labels
        try:
            s_true = silhouette_score(X, labels_true)
        except Exception:
            s_true = np.nan
        ss_tl_list.append(s_true)

        # horizontal split at midpoint of Y
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        y_mid = (y_min + y_max) / 2.0
        labels_h = (X[:, 1] > y_mid).astype(int)

        if len(np.unique(labels_h)) < 2:
            s_h = np.nan
        else:
            try:
                s_h = silhouette_score(X, labels_h)
            except Exception:
                s_h = np.nan
        ss_hl_list.append(s_h)

        datasets.append((X, labels_h, labels_true, y_mid, int(size_2), ir))

    imbalance_arr = np.asarray(imbalance_list)
    ss_tl_arr = np.asarray(ss_tl_list)
    ss_hl_arr = np.asarray(ss_hl_list)

    # pick example where horizontal > true, otherwise pick the max horizontal
    better_idx = np.where(ss_hl_arr > ss_tl_arr)[0]
    chosen_idx = int(better_idx[0]) if better_idx.size > 0 else int(np.nanargmax(ss_hl_arr))

    X_plot, horiz_labels, true_labels, y_mid, size_val, ir_val = datasets[chosen_idx]

    # ---- Plot combined figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Silhouette vs Imbalance Ratio
    axes[0].plot(imbalance_arr, ss_tl_arr, marker='o', linewidth=2,
                 label='Silhouette (true labels, 3 clusters)')
    axes[0].plot(imbalance_arr, ss_hl_arr, marker='s', linewidth=2,
                 label='Silhouette (horizontal split, 2 labels)')
    axes[0].axvline(x=imbalance_arr[chosen_idx], color='gray', linestyle='--', alpha=0.6)
    axes[0].text(imbalance_arr[chosen_idx], np.nanmax([np.nanmax(ss_tl_arr), np.nanmax(ss_hl_arr)]) * 0.9,
                 f"shown example\nIR={imbalance_arr[chosen_idx]:.2f}",
                 ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    axes[0].set_xlabel('Imbalance Ratio (IR)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Silhouette vs Imbalance Ratio', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Middle: Horizontal-split labeling
    label_colors_h = [LABEL_COLOR_MAP[i] for i in horiz_labels]
    axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=label_colors_h, s=25, alpha=0.7)
    axes[1].axhline(y=y_mid, color='black', linestyle='--', linewidth=2, alpha=0.8)
    axes[1].set_title(
        f'Horizontal split labels\nn_minority={size_val}, IR={ir_val:.2f}\n'
        f'Sil(HL)={ss_hl_arr[chosen_idx]:.3f} vs Sil(TL)={ss_tl_arr[chosen_idx]:.3f}',
        fontsize=11, fontweight='bold'
    )
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(alpha=0.3)

    # Right: True labels
    label_colors_t = [LABEL_COLOR_MAP[i] for i in true_labels]
    axes[2].scatter(X_plot[:, 0], X_plot[:, 1], c=label_colors_t, s=25, alpha=0.7)
    axes[2].set_title('True labels', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_filename)

    return imbalance_arr, ss_tl_arr, ss_hl_arr


# -------------------- Main --------------------

def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("ANALYSIS 1: CLUSTER OVERLAP VS CENTROID DISTANCE")
    print("=" * 60)
    centroid_dists, overlap_ratios = analyze_cluster_overlap(
        distances=np.linspace(6, 1, 20),
        n_per_cluster=300,
        cluster_std=1.0,
        save_prefix="analysis_overlap",
        rng=rng,
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
        save_prefix="analysis_imbalance",
        rng=rng,
    )
    print(f"Minority cluster size range: {minority_sizes[0]} to {minority_sizes[-1]}")
    print(f"Imbalance ratio range: {imb_ratios[0]:.2f} to {imb_ratios[-1]:.2f}")
    pct2 = safe_percent_change(imb_ratios[0], imb_ratios[-1])
    if pct2 is None:
        print("Cannot compute percent change for imbalance (initial value is 0).")
    else:
        print(f"As minority cluster shrinks, imbalance increases by {pct2:.1f}%")


    print("\n" + "=" * 60)
    print("ANALYSIS 3: OVERLAP VS SILHOUETTE")
    print("=" * 60)
    analyze_overlap_silhouette(rng=rng)

    print("\n" + "=" * 60)
    print("ANALYSIS 4: IMBALANCE VS SILHOUETTE")
    print("=" * 60)
    analyze_imbalance_silhouette(rng=rng)







if __name__ == "__main__":
    main()
