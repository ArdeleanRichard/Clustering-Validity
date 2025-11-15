import numpy as np
import pandas as pd
import time
from tabulate import tabulate

from constants import METRICS, MAP_LOWER_IS_BETTER, MAP_METRIC_TO_FUNCTION

def get_metric(name):
    """
    Returns a math function from the math library based on the given name.
    """
    func = MAP_METRIC_TO_FUNCTION.get(name, lambda *args, **kwargs: f"No function named '{name}'")

    # Wrap it to accept any arguments safely
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def choose_metric(metric, data, labels, *args, **kwargs):
    metric_function = get_metric(metric)
    return metric_function(data, labels, *args, **kwargs)



def create_metric_table(X, label_sets=None, metrics=METRICS, decimals=3, save="metrics.csv", prnt=True):
    """
    Computes clustering metrics for multiple label sets and creates a table.

    Parameters:
    - X: np.array, the data.
    - gt: np.array, ground truth labels.
    - choose_metric: function(metric, data, labels) -> float, returns metric value.
    - label_sets: dict, optional, keys are label names and values are label arrays.
                  Default: {'gt': gt, 'dp': dp, 'vl': vl, 'hl': hl, 'rl': rl}
    - metrics: list of str, metric names to compute.
    - csv_path: str, file path to save CSV.

    Returns:
    - df: pd.DataFrame containing metrics.
    """

    if label_sets is None:
        raise ValueError("Please provide a dictionary of label sets.")

    # Create a DataFrame to store results
    table = pd.DataFrame(index=metrics, columns=label_sets.keys())

    # Compute each metric for each label set
    for metric in metrics:
        for label_name, labels in label_sets.items():
            try:
                table.loc[metric, label_name] = choose_metric(metric=metric, data=X, labels=labels)
            except Exception as e:
                table.loc[metric, label_name] = np.nan
                print(f"Warning: Metric {metric} failed for {label_name}: {e}")

    if prnt is not None:
        # # Print table nicely with tabs
        # print("\t" + "\t".join(table.columns))
        # for metric in table.index:
        #     values = "\t".join([f"{table.loc[metric, col]:.4f}" if pd.notna(table.loc[metric, col]) else "nan"
        #                         for col in table.columns])
        #     print(f"{metric}\t{values}")
        rows = []
        for metric in table.index:
            row = [metric]
            for i, col in enumerate(table.columns):
                v = table.loc[metric, col]

                if pd.notna(v):
                    cell = f"{v:.{decimals}f}"
                else:
                    cell = "nan"
                # Append "s" to the last column (time)
                if i == len(table.columns) - 1:
                    cell += "s"
                row.append(cell)
            rows.append(row)
        headers = ["metric"] + list(table.columns)
        print(tabulate(rows, headers=headers, tablefmt="plain", stralign="right", numalign="right"))


    if save is not None:
        table.to_csv(save)

    return table


def create_metric_table_with_arrows(X, label_sets=None, metrics=METRICS, decimals=3, save="metrics.csv", prnt=True):
    def is_metric_reversed(metric):
        """
        Returns boolean indicating whether higher (↑) or lower (↓) is better.
        """
        # Normalize metric name to lowercase for comparison
        return True if metric.lower() in MAP_LOWER_IS_BETTER else False

    if label_sets is None:
        raise ValueError("Please provide a dictionary of label sets.")

    # 1. Compute metrics
    table = {}
    for metric in metrics:
        table[metric] = {}

        time_values = []
        time_count = len(label_sets.items())

        for label_name, labels in label_sets.items():
            try:
                time_start = time.time()
                val = choose_metric(metric=metric, data=X, labels=labels)
                time_end = time.time()
                time_values.append(time_end-time_start)

                if type(val) == np.ndarray:
                    print(f"ERROR: {metric} gave np.array {val}")
            except Exception as e:
                print(f"Warning: Metric {metric} failed for {label_name}: {e}")
                val = np.nan
            table[metric][label_name] = round(val, decimals) if not np.isnan(val) else np.nan
        table[metric]["time"] = sum(time_values) / time_count

    table = pd.DataFrame(table).T  # metrics as rows
    table = table[list(label_sets.keys()) + ["time"]]  # ensure correct column order

    # 2. Print with arrows if requested
    if prnt:
        # print("\t" + "\t".join(table.columns))
        # for metric in table.index:
        #     arrow = " (↓)" if is_metric_reversed(metric) else " (↑)"
        #     values = "\t".join(
        #         [f"{table.loc[metric, col]:.{decimals}f}" if pd.notna(table.loc[metric, col]) else "nan"
        #          for col in table.columns])
        #     print(f"{metric}{arrow}\t{values}s") # last values on row is time
        rows = []
        for metric in table.index:
            arrow = " (↓)" if is_metric_reversed(metric) else " (↑)"
            row = [metric + arrow]
            for i, col in enumerate(table.columns):
                v = table.loc[metric, col]

                if pd.notna(v):
                    cell = f"{v:.{decimals}f}"
                else:
                    cell = "nan"
                # Append "s" to the last column (time)
                if i == len(table.columns) - 1:
                    cell += "s"
                row.append(cell)
            rows.append(row)
        headers = ["metric"] + list(table.columns)
        print(tabulate(rows, headers=headers, tablefmt="plain", stralign="right", numalign="right"))

    # 3. Build CSV table with arrows in index and * for worse values
    csv_data = {}

    for metric in metrics:
        # Add arrow to metric name
        lower_better = is_metric_reversed(metric)
        arrow = " (↓)" if lower_better else " (↑)"

        metric_with_arrow = f"{metric}{arrow}"

        values = table.loc[metric].values

        # Reference value is the FIRST column (gt column)
        gt_val = values[0]
        time_val = values[-1]

        row_dict = {}
        for col_name, val in zip(label_sets.keys(), values):
            if np.isnan(val):
                val_str = "nan"
            else:
                val_str = f"{val:.{decimals}f}"
                # Add asterisk for columns other than the first (gt)
                if col_name != list(label_sets.keys())[0] and not np.isnan(gt_val):
                    if (lower_better and val < gt_val) or (not lower_better and val > gt_val):
                        val_str += "*"
            row_dict[col_name] = val_str

        row_dict["time"] = f"{time_val:.{decimals}f}s"

        csv_data[metric_with_arrow] = row_dict

    # Create DataFrame properly
    result_table = pd.DataFrame.from_dict(csv_data, orient='index')

    if save is not None:
        result_table.to_csv(save, index=True, encoding="utf-8")

    return result_table
