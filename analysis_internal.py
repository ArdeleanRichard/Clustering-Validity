import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from analysis_measures import MAP_MEASURES
from constants import LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS_INTERNAL
from constants_maps import MAP_INTERNAL_METRICS
from load_datasets import generate_clusters_analysis



# -------------------- Plot helpers --------------------

def plot_analysis(cvi_str, measure_str, labelset_str, measure_arr, cvi_tl_arr, cvi_fl_arr, chosen_idx, datasets, save_filename):
    cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]
    measure_name_acronym, measure_name_full, measure_function = MAP_MEASURES[measure_str]
    val_name = MAP_MEASURE_TO_VARIABLE[measure_str]
    labelset_name, labelset, line_func = MAP_LABELSET[labelset_str]

    X_plot, false_labels, true_labels, y_mid, val, measure_val = datasets[chosen_idx]

    # ---- Plot combined figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Silhouette vs Imbalance Ratio
    axes[0].plot(measure_arr, cvi_tl_arr, marker='o', linewidth=2, label=f'{cvi_name_full} (True labels, 3 clusters)')
    axes[0].plot(measure_arr, cvi_fl_arr, marker='s', linewidth=2, label=f'{cvi_name_full} ({labelset_name} labels, 2 labels)')
    axes[0].axvline(x=measure_arr[chosen_idx], color='gray', linestyle='--', alpha=0.6)
    axes[0].text(measure_arr[chosen_idx], np.nanmax([np.nanmax(cvi_tl_arr), np.nanmax(cvi_fl_arr)]) * 0.9,
                 f"shown example\n{measure_name_acronym}={measure_arr[chosen_idx]:.2f}",
                 ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    axes[0].set_xlabel(f'{measure_name_full} ({measure_name_acronym})', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'{cvi_name_full}', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{cvi_name_full} vs {measure_name_full}', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Line-split labeling
    labelset_false_colors = [LABEL_COLOR_MAP[i] for i in false_labels]
    axes[1].scatter(X_plot[:, 0], X_plot[:, 1], c=labelset_false_colors, s=25, alpha=0.7)
    p1, p2 = line_func(X)  # Get the two points for plotting
    axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=2, alpha=0.8)  # general line
    axes[1].set_title(
        f'{labelset_name} split labels ({labelset_str.upper()}) \n{val_name}={val}, {measure_name_acronym}={measure_val:.2f}\n'
        f'{cvi_name_full} ({labelset_str.upper()})={cvi_fl_arr[chosen_idx]:.3f} vs {cvi_name_full} (TL)={cvi_tl_arr[chosen_idx]:.3f}',
        fontsize=11, fontweight='bold'
    )
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(alpha=0.3)

    # Right: True labels
    label_colors_t = [LABEL_COLOR_MAP[i] for i in true_labels]
    axes[2].scatter(X_plot[:, 0], X_plot[:, 1], c=label_colors_t, s=25, alpha=0.7)
    axes[2].set_title(f'True labels (TL) \n{val_name}={val:.2f}, {measure_name_acronym}={measure_arr[chosen_idx]:.2f}', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, save_filename)


def save_fig(fig, filename, dpi=300):
    fig.savefig(f"./{FOLDER_FIGS_ANALYSIS_INTERNAL}/{filename}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: ./{FOLDER_FIGS_ANALYSIS_INTERNAL}/{filename}.png")


def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })




def generate_analysis_datasets(
        measure_str,
        n_per_cluster=300,
        cluster_std=1.0,
        centers=None,
        cluster_2_sizes=None,
        distances=None,

):
    datasets = []
    if measure_str == "imbalance":
        if cluster_2_sizes is None:
            cluster_2_sizes = np.linspace(n_per_cluster, int(n_per_cluster * 0.1), 20).astype(int)
        if centers is None:
            centers = [(0, 0), (5, 0), (2.5, 4)]
        for size_2 in cluster_2_sizes:
            X, labels_true = generate_clusters_analysis(centers, [n_per_cluster, n_per_cluster, int(size_2)], cluster_std=cluster_std)
            datasets.append((X, labels_true, size_2))

    if measure_str == "overlap":
        if distances is None:
            distances = np.linspace(6, 1, 20)

        for d in distances:
            centers = [(0.0, 0.0), (d, 0.0), (3.0, 5.0)]
            X, labels_true = generate_clusters_analysis(centers, [n_per_cluster, n_per_cluster, n_per_cluster], cluster_std=cluster_std)
            datasets.append((X, labels_true, d))

    return datasets


MAP_MEASURE_TO_VARIABLE = {
    "imbalance": "n_minority",
    "overlap": "distance"
}



MAP_LABELSET = {
    "hl": (
        "Horizontal",
        lambda X: (X[:, 1] > ((X[:, 1].min() + X[:, 1].max()) / 2.0)).astype(int),
        lambda X: ((X[:, 0].min(), (X[:, 1].min() + X[:, 1].max()) / 2.0), (X[:, 0].max(), (X[:, 1].min() + X[:, 1].max()) / 2.0))
    ),
    "vl": (
        "Vertical",
        lambda X: (X[:, 0] > ((X[:, 0].min() + X[:, 0].max()) / 2.0)).astype(int),
        lambda X: (((X[:, 0].min() + X[:, 0].max()) / 2.0, X[:, 1].min()), ((X[:, 0].min() + X[:, 0].max()) / 2.0, X[:, 1].max())),
    ),
}



def analyze_measure(
    cvi_str,
    measure_str,
    labelset_str,
    save_filename="analysis"
):
    """
    Vary the minority cluster size and compute:
      - Imbalance Ratio (IR)
      - Silhouette for true labels (3 clusters)
      - Silhouette for horizontal midpoint split (2 labels)
    Plot Silhouette vs IR and display an example where the horizontal split outperforms
    the true labeling (or the horizontal split with max silhouette if none).
    Returns (measure_arr, cvi_tl_arr, cvi_hl_arr).
    """
    cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]
    measure_name_acronym, measure_name_full, measure_function = MAP_MEASURES[measure_str]

    set_plot_style()

    datasets = generate_analysis_datasets(measure_str)

    measure_list = []
    cvi_tl_list = []
    cvi_hl_list = []
    datasets_infos = []

    for X, labels_true, val in datasets:
        measure_score = measure_function(X, labels_true)
        measure_list.append(measure_score)

        # silhouette for true labels
        try:
            cvi_true = cvi_function(X, labels_true)
        except Exception:
            cvi_true = np.nan
        cvi_tl_list.append(cvi_true)

        # split by horizontal/vertical
        labels_false = MAP_LABELSET[measure_str]

        if len(np.unique(labels_false)) < 2:
            cvi_horiz = np.nan
        else:
            try:
                cvi_horiz = silhouette_score(X, labels_false)
            except Exception:
                cvi_horiz = np.nan
        cvi_hl_list.append(cvi_horiz)

        datasets_infos.append((X, labels_false, labels_true, y_mid, val, measure_score))

    measure_arr = np.asarray(measure_list)
    cvi_tl_arr = np.asarray(cvi_tl_list)
    cvi_fl_arr = np.asarray(cvi_hl_list)

    # pick example where horizontal > true, otherwise pick the max horizontal
    better_idx = np.where(cvi_fl_arr > cvi_tl_arr)[0]
    chosen_idx = int(better_idx[0]) if better_idx.size > 0 else int(np.nanargmax(cvi_fl_arr))

    X_plot, horiz_labels, true_labels, y_mid, size_val, ir_val = datasets_infos[chosen_idx]

    plot_analysis(cvi_str,
                  measure_str,
                  measure_arr,
                  cvi_tl_arr,
                  cvi_fl_arr,

                  chosen_idx=chosen_idx,
                  datasets=datasets_infos,

                  save_filename=f"{save_filename}_{measure_str}_{labelset_str}_{cvi_str}")


    return measure_arr, cvi_tl_arr, cvi_fl_arr


# -------------------- Main --------------------

def main():
    print("\n" + "=" * 60)
    print("ANALYSIS: Overlap vs SILHOUETTE")
    print("=" * 60)
    analyze_measure("silhouette", "overlap")

    print("\n" + "=" * 60)
    print("ANALYSIS: Imbalance vs SILHOUETTE")
    print("=" * 60)
    analyze_measure("silhouette", "imbalance")


if __name__ == "__main__":
    main()
