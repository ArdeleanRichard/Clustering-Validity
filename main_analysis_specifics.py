from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from constants import FOLDER_RESULTS_CACHE, FOLDER_RESULTS_SPECIFICS

# ── Publication-quality global style ─────────────────────────────────────────
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


def _style_ax(ax, values, metric, di, subplot_row, n_cols, color,
              is_top_row_of_group=False):
    """
    Shared axis styling: x-tick labels, y-axis, spines, grid.

    Parameters
    ----------
    ax               : Axes
    values           : array-like  — the bar values (used for x-tick count)
    metric           : str
    di               : int  — dataset (column) index
    subplot_row      : int  — absolute subplot row index
    n_cols           : int
    color            : str  — bar fill colour
    is_top_row_of_group : bool — True for the first metric row of a clusterer block
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
        ax.set_ylabel(metric, fontsize=12, fontweight="bold", labelpad=6,
                      fontfamily="serif")


def plot_all_in_one(plot_data, datasets, clusterers, cvis, file_name):
    out_dir = Path(FOLDER_RESULTS_SPECIFICS)

    all_metrics = ["ARI"] + list(cvis)
    n_metrics   = len(all_metrics)
    n_clusterers = len(clusterers)
    n_cols      = len(datasets)

    # ── colour palette ───────────────────────────────────────────────────────
    metric_colors = {"ARI": "#4D4D4D"}
    cvi_palette   = ["#2166AC", "#C0392B", "#27AE60", "#D35400", "#6C3483"]
    for i, cvi in enumerate(cvis):
        metric_colors[cvi] = cvi_palette[i % len(cvi_palette)]

    # ── GridSpec layout ──────────────────────────────────────────────────────
    # Between each pair of clusterer groups we insert a thin spacer row that
    # will hold the clusterer label for the group *below* it (except for the
    # very first group whose label sits above its first data row).
    #
    # GridSpec row structure (top → bottom):
    #   [label row for clusterer 0]          ← spacer, height_ratio = LABEL
    #   [data rows for clusterer 0]          ← n_metrics rows, ratio = DATA each
    #   [label row for clusterer 1]          ← spacer
    #   [data rows for clusterer 1]
    #   …
    #
    DATA  = 3.0   # relative height of one data row (inches equivalent)
    LABEL = 0.45  # relative height of one label/spacer row

    # Build the height-ratio list
    height_ratios = []
    for ci in range(n_clusterers):
        height_ratios.append(LABEL)                # label spacer row
        height_ratios.extend([DATA] * n_metrics)   # data rows

    n_gs_rows = len(height_ratios)   # = n_clusterers * (1 + n_metrics)

    col_width  = max(5, len(datasets) * 0.6)
    # Total figure height: sum of all ratios scaled to inches
    fig_height = sum(height_ratios)
    fig_width  = n_cols * col_width

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = fig.add_gridspec(
        n_gs_rows, n_cols,
        height_ratios=height_ratios,
        hspace=0.55,   # vertical gap between all rows
        wspace=0.35,   # horizontal gap between columns
    )

    for ci, clusterer in enumerate(clusterers):
        # GridSpec row index of this clusterer's label row
        label_gs_row = ci * (1 + n_metrics)

        # ── clusterer label in the spacer row ────────────────────────────────
        # Span all columns so the text is centred over the full width.
        ax_label = fig.add_subplot(gs[label_gs_row, :])
        ax_label.set_axis_off()
        ax_label.text(
            0.5, 0.5,
            clusterer,
            transform=ax_label.transAxes,
            fontsize=13,
            fontweight="bold",
            fontfamily="serif",
            va="center",
            ha="center",
        )

        for mi, metric in enumerate(all_metrics):
            data_gs_row = label_gs_row + 1 + mi   # skip the label row

            for di, data in enumerate(datasets):
                ax = fig.add_subplot(gs[data_gs_row, di])

                df = plot_data[clusterer][data]

                if df is None or metric not in df.index:
                    ax.set_visible(False)
                    continue

                values = df.loc[metric].values
                color  = metric_colors.get(metric, "#333333")

                _style_ax(ax, values, metric, di, data_gs_row, n_cols, color)

                # Dataset name as column header — only on very first data row
                # of the entire figure (first clusterer, first metric)
                if ci == 0 and mi == 0:
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

                _style_ax(ax, values, metric, ci, mi, n_cols, color)

                # Clusterer name as column header (top row only)
                if mi == 0:
                    ax.set_title(clusterer, fontsize=13, fontweight="bold", pad=8,
                                 fontfamily="serif")

        fig.suptitle(
            f"{data} — ARI + CVIs",
            fontsize=16,
            fontweight="bold",
            y=1.002,
            fontfamily="serif",
        )

        plt.tight_layout(h_pad=0.8, w_pad=0.8)

        out_fig = out_dir / f"{file_name}_{data}.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    datasets = ["coil20", "olivetti", "yaleA"]
    clusterers = ["AgglomerativeClustering", "HDBSCAN", "KMeans", "SpectralClustering"]
    cvis = ["DBCV", "AD-S", "AD-idea"]
    plot_data = aggregate(datasets, clusterers, cvis)
    plot_all_in_one(plot_data, datasets, clusterers, cvis,  file_name="analysis_data_image")
    plot_per_dataset(plot_data, datasets, clusterers, cvis, file_name="analysis_data_image")

    datasets = ["ecoli", "glass", "ionosphere", "sonar", "statlog", "wdbc", "wine", "yeast"]
    clusterers = ["AgglomerativeClustering", "HDBSCAN", "KMeans", "MeanShift", "SpectralClustering"]
    cvis = ["VIASCKDE", "COP", "CS", "SD", "CH", "AD-CH", "AD-idea"]
    plot_data = aggregate(datasets, clusterers, cvis)
    plot_all_in_one(plot_data, datasets, clusterers, cvis,  file_name="analysis_data_real")
    plot_per_dataset(plot_data, datasets, clusterers, cvis, file_name="analysis_data_real")