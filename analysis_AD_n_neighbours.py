import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from constants import LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS_AD
from constants_maps import MAP_INTERNAL_METRICS
from load_datasets import create_data1


def plot_n_neighbours_analysis(
        cvi_str,
        X,
        labels,
        ks=range(1, 11)
):
    """

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    labels : array-like, shape (n_samples,)
    ks : iterable of int
        Values of k to evaluate (converted to list).

    """
    cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]

    X = np.asarray(X)
    labels = np.asarray(labels)
    ks = list(ks)

    fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(10, 4))
    label_color = [LABEL_COLOR_MAP[i] for i in labels]
    ax_scatter.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)


    scores = []
    for k in ks:
        val = cvi_function(X, labels, n_neighbors=k)

        scores.append(float(val))
    scores = np.array(scores)
    ax_curve.plot(ks, scores, marker='o', label=cvi_name_full)

    ax_curve.set_xlabel("k")
    ax_curve.set_xticks(ks)
    ax_curve.set_title(rf"{cvi_name_full} vs $\it{{n\_neighbours}}$ parameter")
    ax_curve.grid(True, linestyle="--", alpha=0.4)
    ax_curve.legend(fontsize="small", loc="best")

    plt.tight_layout()

    plt.savefig(FOLDER_FIGS_ANALYSIS_AD+f"{cvi_name_full}.png")
    plt.close()




def plot_n_neighbours_analysis_all(
        X,
        labels,
        ks=range(1, 11)
):
    """

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    labels : array-like, shape (n_samples,)
    ks : iterable of int
        Values of k to evaluate (converted to list).

    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    ks = list(ks)

    import matplotlib as mpl
    # Local rc params so we don't globally change user settings
    rc = {
        "figure.figsize": (20, 12),
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
    }
    mpl.rcParams.update(rc)
    fig, axes = plt.subplots(2, 3, figsize=rc["figure.figsize"])
    fig.patch.set_facecolor("white")
    fig.delaxes(axes[0, 2])

    label_color = [LABEL_COLOR_MAP[i] for i in labels]
    axes[0, 0].scatter(
        X[:, 0],
        X[:, 1],
        c=label_color,
        marker='o',
        edgecolors='k',
        linewidths=0.8,
        alpha=0.95,
        s=50,
        rasterized=False
    )
    axes[0, 0].set_title("D1 with ground truth labels")
    axes[0, 0].set_axisbelow(True)
    axes[0, 0].grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    cvi_strs = [
        "ad_silhouette",
        "ad_db",
        "ad_ch",
        "ad_idea",
    ]
    MAP_CVI_TO_SCORES = {}
    for cvi_str in cvi_strs:
        cvi_name_full, cvi_function = MAP_INTERNAL_METRICS[cvi_str]

        scores = []
        for k in ks:
            val = cvi_function(X, labels, n_neighbors=k)
            scores.append(float(val))
        scores = np.array(scores)
        MAP_CVI_TO_SCORES[cvi_str] = (cvi_name_full, scores)

    # helper to style line plots and ensure a visible point for every value
    def style_line_axis(ax, x, y, label_text):
        # main line with markers
        ax.plot(
            x,
            y,
            marker='o',
            markersize=8,
            markeredgewidth=0.9,
            linewidth=2.0,
            markeredgecolor='k',
            label=label_text,
            zorder=2
        )
        # explicit scatter overlay to make each point visually stronger
        ax.scatter(
            x,
            y,
            s=50,
            edgecolors='k',
            linewidths=0.9,
            zorder=3
        )
        ax.set_xlabel(rf"$\it{{n\_neighbours}}$")
        # ax.set_xticks(ks)
        step = max(1, len(ks) // 6)
        ax.set_xticks(ks[::step])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
        ax.legend(frameon=False, fontsize=mpl.rcParams["legend.fontsize"], loc="best")
        ax.tick_params(axis='both', which='major', length=6)

    style_line_axis(axes[0, 1], ks, MAP_CVI_TO_SCORES["ad_idea"][1], MAP_CVI_TO_SCORES["ad_idea"][0])
    axes[0, 1].set_title(rf'{MAP_CVI_TO_SCORES["ad_idea"][0]} vs $\it{{n\_neighbours}}$ parameter')

    style_line_axis(axes[1, 0], ks, MAP_CVI_TO_SCORES["ad_silhouette"][1], MAP_CVI_TO_SCORES["ad_silhouette"][0])
    axes[1, 0].set_title(rf'{MAP_CVI_TO_SCORES["ad_silhouette"][0]} vs $\it{{n\_neighbours}}$ parameter')

    style_line_axis(axes[1, 1], ks, MAP_CVI_TO_SCORES["ad_db"][1], MAP_CVI_TO_SCORES["ad_db"][0])
    axes[1, 1].set_title(rf'{MAP_CVI_TO_SCORES["ad_db"][0]} vs $\it{{n\_neighbours}}$ parameter')

    style_line_axis(axes[1, 2], ks, MAP_CVI_TO_SCORES["ad_ch"][1], MAP_CVI_TO_SCORES["ad_ch"][0])
    axes[1, 2].set_title(rf'{MAP_CVI_TO_SCORES["ad_ch"][0]} vs $\it{{n\_neighbours}}$ parameter')

    plt.tight_layout(pad=2.0)

    # save with higher DPI so text and markers are crisp
    plt.savefig(FOLDER_FIGS_ANALYSIS_AD + f"{cvi_name_full}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    X, y_true = create_data1(1000)
    ks = range(3, 50)

    plot_n_neighbours_analysis("ad_silhouette", X, y_true, ks=ks)
    plot_n_neighbours_analysis_all(X, y_true, ks=ks)