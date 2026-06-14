import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from constants import scale
from constants_maps import CVIs
from load_CVIs import choose_CVI
from load_datasets import create_data5


def _run_timing_loop(axis_values, axis_label, fixed_label, metrics, n_runs, output_csv_prefix):
    """
    Core timing loop shared by both analyses.

    Parameters
    ----------
    axis_values : list
        Values swept on the varying axis (ns or ds).
    axis_label : str
        Column label key, either 'n' or 'd'.
    fixed_label : str
        Human-readable description of the fixed dimension, e.g. 'n=1000' or 'd=2'.
    metrics : list
        CVI metric names to time.
    n_runs : int
        Number of repetitions per (metric, configuration).
    output_csv_prefix : str
        Prefix for the two output CSV files.

    Returns
    -------
    timing_mean, timing_std : pd.DataFrame
        Rows = metrics, columns = axis values.
    """
    col_labels = [f"{axis_label}={v}" for v in axis_values]
    timing_mean = pd.DataFrame(index=metrics, columns=col_labels, dtype=float)
    timing_std  = pd.DataFrame(index=metrics, columns=col_labels, dtype=float)

    for v in axis_values:
        col = f"{axis_label}={v}"
        n = v if axis_label == 'n' else int(fixed_label.split('=')[1])
        d = v if axis_label == 'd' else int(fixed_label.split('=')[1])

        print(f"\n{'='*60}")
        print(f"Varying {axis_label}={v}  (fixed: {fixed_label})")
        print(f"{'='*60}")

        X_raw, labels = create_data5(n, d)
        X = MinMaxScaler(scale).fit_transform(X_raw)

        for metric in metrics:
            run_times = []
            failed = False

            for run in range(n_runs):
                t0 = time.time()
                try:
                    _ = choose_CVI(cvi=metric, data=X, labels=labels)
                    run_times.append(time.time() - t0)
                except Exception as e:
                    print(f"  [{metric}] run {run+1} FAILED: {e}")
                    failed = True
                    break

            if failed or len(run_times) == 0:
                mean_t, std_t = np.nan, np.nan
            else:
                mean_t = np.mean(run_times)
                std_t  = np.std(run_times)

            timing_mean.loc[metric, col] = mean_t
            timing_std.loc[metric,  col] = std_t
            print(f"  {metric:20s}: {mean_t:.4f}s ± {std_t:.4f}s  (over {n_runs} runs)")

    mean_path = f"{output_csv_prefix}_mean.csv"
    std_path  = f"{output_csv_prefix}_std.csv"
    timing_mean.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN").to_csv(mean_path)
    timing_std.applymap( lambda x: f"{x:.3f}" if pd.notna(x) else "NaN").to_csv(std_path)

    combined = timing_mean.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN") + " ± " + timing_std.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    csv_path = f"{output_csv_prefix}.csv"
    combined.to_csv(csv_path)

    print(f"\n>>> Saved: {mean_path}")
    print(f">>> Saved: {std_path}")

    return timing_mean, timing_std


def run_timing_vary_samples(
    ns=[500, 1000, 2000, 5000, 10000, 15000, 20000],
    fixed_d=2,
    n_runs=3,
    metrics=None,
    output_csv_prefix="timing_cvi_vary_n",
):
    """
    Sweep over sample sizes (n) with a fixed number of dimensions.
    Rows = metrics, columns = n values.
    """
    if metrics is None:
        metrics = CVIs
    print(f"\n{'#'*60}")
    print(f"# TIMING ANALYSIS: varying n, fixed d={fixed_d}")
    print(f"{'#'*60}")
    return _run_timing_loop(
        axis_values=ns,
        axis_label='n',
        fixed_label=f"d={fixed_d}",
        metrics=metrics,
        n_runs=n_runs,
        output_csv_prefix=output_csv_prefix,
    )


def run_timing_vary_dimensions(
    ds=[2, 5, 10, 20, 50, 100, 200, 500],
    fixed_n=1000,
    n_runs=3,
    metrics=None,
    output_csv_prefix="timing_cvi_vary_d",
):
    """
    Sweep over dimensionality (d) with a fixed number of samples.
    Rows = metrics, columns = d values.
    """
    if metrics is None:
        metrics = CVIs
    print(f"\n{'#'*60}")
    print(f"# TIMING ANALYSIS: varying d, fixed n={fixed_n}")
    print(f"{'#'*60}")
    return _run_timing_loop(
        axis_values=ds,
        axis_label='d',
        fixed_label=f"n={fixed_n}",
        metrics=metrics,
        n_runs=n_runs,
        output_csv_prefix=output_csv_prefix,
    )


if __name__ == "__main__":
    n_runs = 5

    # --- Analysis 1: vary number of samples, d fixed at 2 ---
    mean_n, std_n = run_timing_vary_samples(
        ns=[500, 1000, 2000, 5000, 10000, 15000, 20000],
        fixed_d=2,
        n_runs=n_runs,
        metrics=CVIs,
        output_csv_prefix="./results/supplementary/timing/timing_cvi_vary_n",
    )

    # --- Analysis 2: vary dimensionality, n fixed at 1000 ---
    CVIs.remove("CDbw")
    mean_d, std_d = run_timing_vary_dimensions(
        ds=[2, 5, 10, 20, 50, 100, 200, 500],
        fixed_n=1000,
        n_runs=n_runs,
        metrics=CVIs,
        output_csv_prefix="./results/supplementary/timing/timing_cvi_vary_d",
    )

    print("\n\n=== VARY N — MEAN TIMING (seconds) ===")
    print(mean_n.to_string())
    print("\n=== VARY D — MEAN TIMING (seconds) ===")
    print(mean_d.to_string())