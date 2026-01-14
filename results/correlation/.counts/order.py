import pandas as pd
from constants_maps import METRICS


def reorder(filename):
    # Load CSV
    df = pd.read_csv(filename, header=None, skiprows=3)

    # Manually set column names
    df.columns = [
        "metric",
        "Agglomerative_correct",
        "DBSCAN_correct",
        "HDBSCAN_correct",
        "KMeans_correct",
        "MeanShift_correct",
        "Spectral_correct",
        "Agglomerative_errors",
        "DBSCAN_errors",
        "HDBSCAN_errors",
        "KMeans_errors",
        "MeanShift_errors",
        "Spectral_errors",
    ]

    # Convert column to ordered categorical
    df["metric"] = pd.Categorical(df["metric"], categories=METRICS, ordered=True)

    # Sort by that column
    df_sorted = df.sort_values("metric")

    # Save result
    df_sorted.to_csv(f"{filename}", index=False)


if __name__ == "__main__":
    reorder("realdata_best_match_by_algorithm.csv")
    reorder("synthdata_best_match_by_algorithm.csv")
    # reorder("realdata_best_match_by_dataset.csv")
    # reorder("synthdata_best_match_by_dataset.csv")