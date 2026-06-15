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
    n_rows      = len(clusterers) * n_metrics
    n_cols      = len(datasets)

    # ── figure dimensions ────────────────────────────────────────────────────
    col_width  = max(5, len(datasets) * 0.6)
    row_height = 3.0

    fig_width  = n_cols * col_width
    fig_height = n_rows * row_height

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # ── colour palette ───────────────────────────────────────────────────────
    # Desaturated, print-safe colours that survive greyscale printing.
    metric_colors = {"ARI": "#4D4D4D"}
    cvi_palette   = ["#2166AC", "#C0392B", "#27AE60", "#D35400", "#6C3483"]
    for i, cvi in enumerate(cvis):
        metric_colors[cvi] = cvi_palette[i % len(cvi_palette)]

    for ci, clusterer in enumerate(clusterers):
        for mi, metric in enumerate(all_metrics):
            subplot_row = ci * n_metrics + mi

            for di, data in enumerate(datasets):
                ax = axes[subplot_row][di]

                df = plot_data[clusterer][data]

                if df is None or metric not in df.index:
                    ax.set_visible(False)
                    continue

                values = df.loc[metric].values
                color  = metric_colors.get(metric, "#333333")

                _style_ax(ax, values, metric, di, subplot_row, n_cols, color,
                          is_top_row_of_group=(mi == 0))

                # Dataset name as column header (top row only)
                if subplot_row == 0:
                    ax.set_title(data, fontsize=13, fontweight="bold", pad=8,
                                 fontfamily="serif")

                # Clusterer label on the right side, vertically centred in its block
                if di == n_cols - 1 and mi == n_metrics // 2:
                    ax.annotate(
                        clusterer,
                        xy=(1.03, 0.5),
                        xycoords="axes fraction",
                        fontsize=12,
                        fontweight="bold",
                        va="center",
                        ha="left",
                        rotation=270,
                        fontfamily="serif",
                    )

    # ── horizontal separator lines between clusterer groups ──────────────────
    # We draw them as figure-level lines positioned between the last subplot row
    # of one clusterer and the first subplot row of the next.
    fig.canvas.draw()  # needed to get accurate axes positions

    for ci in range(1, len(clusterers)):
        # Row index of the last metric row in the previous clusterer block
        row_above = ci * n_metrics - 1
        row_below = ci * n_metrics

        # Gather y positions from all columns for both boundary rows
        y_bottoms = []
        y_tops    = []
        for di in range(n_cols):
            ax_above = axes[row_above][di]
            ax_below = axes[row_below][di]
            if not ax_above.get_visible() and not ax_below.get_visible():
                continue
            ax_ref = ax_above if ax_above.get_visible() else ax_below
            bbox_above = ax_above.get_position()
            bbox_below = ax_below.get_position()
            y_bottoms.append(bbox_above.y0)
            y_tops.append(bbox_below.y1)

        if not y_bottoms:
            continue

        # Midpoint between the two groups in figure coordinates
        y_line = (min(y_bottoms) + max(y_tops)) / 2

        line = mpl.lines.Line2D(
            [0.01, 0.99], [y_line, y_line],
            transform=fig.transFigure,
            color="#555555",
            linewidth=1.2,
            linestyle="--",
            zorder=10,
        )
        fig.add_artist(line)

    # ── figure title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "ARI + CVIs",
        fontsize=16,
        fontweight="bold",
        y=1.002,
        fontfamily="serif",
    )

    plt.tight_layout(h_pad=0.8, w_pad=0.8)

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