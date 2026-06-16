from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from constants import FOLDER_RESULTS_CACHE, FOLDER_RESULTS_SPECIFICS

mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.grid":          True,
    "grid.color":         "#CCCCCC",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.6,
    "axes.axisbelow":     True,   # grid behind bars
    "figure.dpi":         150,
    "savefig.dpi":        300,    # high-res for print
})


def aggregate(datasets, clusterers, cvis):
    """
    Reads ARI and CVI result files for every (dataset, clusterer) pair,
    combines them into per-pair CSVs (sorted ascending by ARI), and
    produces a single summary figure for the entire call.

    Figure layout:
        rows    = clusterers  (one row-group per clusterer)
        columns = datasets
        within each cell: one subplot per metric (ARI + each CVI),
                          stacked vertically — so total subplot rows =
                          len(clusterers) * len(all_metrics)

    Parameters
    ----------
    datasets   : list[str]
    clusterers : list[str]
    cvis       : list[str]   CVI row names expected in the metrics CSV
    """
    in_dir = Path(FOLDER_RESULTS_CACHE)
    out_dir = Path(FOLDER_RESULTS_SPECIFICS)

    # ── collect all DataFrames first so we know what exists ──────────────────
    # combined_data[clusterer][dataset] = combined DataFrame (or None)
    combined_data = {cl: {ds: None for ds in datasets} for cl in clusterers}

    for data in datasets:
        for clusterer in clusterers:
            base_name = f"{data}_{clusterer}"

            ari_path = in_dir / f"{base_name}_ARI.csv"
            metrics_path = in_dir / f"{base_name}.csv"

            if not ari_path.exists():
                print(f"Missing ARI file: {ari_path}")
                continue
            if not metrics_path.exists():
                print(f"Missing metrics file: {metrics_path}")
                continue

            ari_df = pd.read_csv(ari_path, index_col=0)
            metrics_df = pd.read_csv(metrics_path, index_col=0)

            ari_rows = ari_df.loc[["ARI"]] if "ARI" in ari_df.index else ari_df

            selected_cvis = [cvi for cvi in cvis if cvi in metrics_df.index]
            cvi_rows = metrics_df.loc[selected_cvis]

            combined = pd.concat([ari_rows, cvi_rows], axis=0)

            if "ARI" in combined.index:
                sorted_columns = combined.loc["ARI"].sort_values(ascending=True).index
                combined = combined[sorted_columns]

            out_csv = out_dir / f"{base_name}_ARI_plus_CVIs.csv"
            combined.to_csv(out_csv)
            print(f"Saved: {out_csv}")

            combined_data[clusterer][data] = combined

    return combined_data


def _style_ax(ax, values, metric, di, color):
    """
    Shared axis styling: x-tick labels, y-axis, spines, grid.

    Parameters
    ----------
    ax               : Axes
    values           : array-like  — the bar values (used for x-tick count)
    metric           : str
    di               : int  — dataset (column) index
    color            : str  — bar fill colour
    """
    n = len(values)
    x = range(n)

    ax.bar(x, values, color=color, width=0.85, linewidth=0, zorder=3)

    ax.set_xlim(-0.5, n - 0.5)

    # ── x-axis: show integer parametrization indices ──────────────────────────
    # For readability, limit to at most ~8 labelled ticks via MaxNLocator.
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=7, integer=True, min_n_ticks=1))
    ax.tick_params(axis="x", labelsize=9, rotation=0, length=3.5)

    # ── y-axis ────────────────────────────────────────────────────────────────
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.get_offset_text().set_fontsize(9)

    # Left y-label: metric name, only in leftmost column
    if di == 0:
        ax.set_ylabel("A" if metric == "AD-idea" else metric, fontsize=12, fontweight="bold", labelpad=6, fontfamily="serif")



def plot_all_in_one(plot_data, datasets, clusterers, cvis, file_name, n_clusterer_cols=1):
    """
    Parameters
    ----------
    n_clusterer_cols : int
        How many clusterer-groups to place side by side.
        e.g. 1 = all stacked vertically (original behaviour)
             2 = two clusterer-groups per row
             3 = three clusterer-groups per row
    """
    out_dir = Path(FOLDER_RESULTS_SPECIFICS)

    all_metrics  = ["ARI"] + list(cvis)
    n_metrics    = len(all_metrics)
    n_clusterers = len(clusterers)
    n_datasets   = len(datasets)

    # ── colour palette ───────────────────────────────────────────────────────
    metric_colors = {"ARI": "#4D4D4D"}
    cvi_palette   = ["#2166AC", "#C0392B", "#27AE60", "#D35400", "#6C3483"]
    for i, cvi in enumerate(cvis):
        metric_colors[cvi] = cvi_palette[i % len(cvi_palette)]

    # ── clusterer layout ─────────────────────────────────────────────────────
    # How many rows of clusterer-groups (ceiling division)
    n_clusterer_rows = (n_clusterers + n_clusterer_cols - 1) // n_clusterer_cols

    # ── GridSpec dimensions ──────────────────────────────────────────────────
    # Each "cell" in the high-level grid is:
    #   rows : 1 label row + n_metrics data rows
    #   cols : n_datasets columns
    #
    # Total GridSpec shape:
    #   gs_rows = n_clusterer_rows * (1 + n_metrics)
    #   gs_cols = n_clusterer_cols  * n_datasets

    DATA  = 3.0
    LABEL = 0.45

    height_ratios = []
    for _ in range(n_clusterer_rows):
        height_ratios.append(LABEL)
        height_ratios.extend([DATA] * n_metrics)

    gs_rows = len(height_ratios)
    gs_cols = n_clusterer_cols * n_datasets

    col_width  = max(5, n_datasets * 0.6)
    fig_height = sum(height_ratios)
    fig_width  = gs_cols * col_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        gs_rows, gs_cols,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.35,
    )

    for ci, clusterer in enumerate(clusterers):
        # Which high-level cell does this clusterer occupy?
        cl_row = ci // n_clusterer_cols   # which row of clusterer-groups
        cl_col = ci % n_clusterer_cols    # which column of clusterer-groups

        # Top-left GridSpec coordinates of this clusterer's block
        gs_row_start = cl_row * (1 + n_metrics)   # label row index
        gs_col_start = cl_col * n_datasets         # first dataset column

        # ── clusterer label spanning its own dataset columns ─────────────────
        ax_label = fig.add_subplot(
            gs[gs_row_start, gs_col_start : gs_col_start + n_datasets]
        )
        ax_label.set_axis_off()
        ax_label.text(
            0.5, 0.5,
            clusterer,
            transform=ax_label.transAxes,
            fontsize=13, fontweight="bold", fontfamily="serif",
            va="center", ha="center",
        )

        for mi, metric in enumerate(all_metrics):
            data_gs_row = gs_row_start + 1 + mi

            for di, data in enumerate(datasets):
                gs_col = gs_col_start + di
                ax = fig.add_subplot(gs[data_gs_row, gs_col])

                df = plot_data[clusterer][data]

                if df is None or metric not in df.index:
                    ax.set_visible(False)
                    continue

                values = df.loc[metric].values
                color  = metric_colors.get(metric, "#333333")

                # di=0 check: only label y-axis for leftmost dataset of this clusterer
                is_leftmost = (di == 0)
                _style_ax(ax, values, metric, 0 if is_leftmost else 1, color)

                # Dataset column header: top metric row, first clusterer-row only
                if mi == 0:
                    ax.set_title(data, fontsize=13, fontweight="bold", pad=8, fontfamily="serif")

    out_fig = out_dir / f"{file_name}.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_fig}")


def plot_per_dataset(plot_data, datasets, clusterers, cvis, file_name):
    out_dir = Path(FOLDER_RESULTS_SPECIFICS)
    all_metrics = ["ARI"] + list(cvis)
    n_metrics   = len(all_metrics)

    # ── colours ───────────────────────────────────────────────────────────────
    metric_colors = {"ARI": "#4D4D4D"}
    cvi_palette   = ["#2166AC", "#C0392B", "#27AE60", "#D35400", "#6C3483"]
    for i, cvi in enumerate(cvis):
        metric_colors[cvi] = cvi_palette[i % len(cvi_palette)]

    n_rows = n_metrics
    n_cols = len(clusterers)

    col_width  = 8
    row_height = 3.0

    for data in datasets:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * col_width, n_rows * row_height),
            squeeze=False,
        )

        for mi, metric in enumerate(all_metrics):
            for ci, clusterer in enumerate(clusterers):
                ax = axes[mi][ci]

                df = plot_data[clusterer][data]

                if df is None or metric not in df.index:
                    ax.set_visible(False)
                    continue

                values = df.loc[metric].values
                color  = metric_colors.get(metric, "#333333")

                _style_ax(ax, values, metric, ci, color)

                # Clusterer name as column header (top row only)
                if mi == 0:
                    ax.set_title(clusterer, fontsize=13, fontweight="bold", pad=8, fontfamily="serif")

        plt.tight_layout(h_pad=0.8, w_pad=0.8)

        out_fig = out_dir / f"{file_name}_{data}.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    datasets = ["coil20", "olivetti", "yaleA"]
    clusterers = ["AgglomerativeClustering", "HDBSCAN", "KMeans", "SpectralClustering"]
    cvis = ["DBCV", "COP", "AD-S", "AD-idea"]
    plot_data = aggregate(datasets, clusterers, cvis)
    plot_all_in_one(plot_data, datasets, clusterers, cvis,  file_name="analysis_data_image", n_clusterer_cols=2)
    plot_per_dataset(plot_data, datasets, clusterers, cvis, file_name="analysis_data_image")

    datasets = ["ecoli", "glass", "ionosphere", "sonar", "statlog", "wdbc", "wine", "yeast"]
    clusterers = ["AgglomerativeClustering", "HDBSCAN", "KMeans", "MeanShift", "SpectralClustering"]
    cvis = ["VIASCKDE", "COP", "CS", "SD", "CH", "AD-CH", "AD-idea"]
    plot_data = aggregate(datasets, clusterers, cvis)
    plot_all_in_one(plot_data, datasets, clusterers, cvis,  file_name="analysis_data_real")
    plot_per_dataset(plot_data, datasets, clusterers, cvis, file_name="analysis_data_real")