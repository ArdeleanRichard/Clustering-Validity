from pathlib import Path
import pandas as pd

from constants import FOLDER_RESULTS_CACHE


def combine_ari_and_cvis(datasets, clusterers, cvis):
    folder = Path(FOLDER_RESULTS_CACHE)

    for data in datasets:
        for clusterer in clusterers:
            base_name = f"{data}_{clusterer}"

            ari_path = folder / f"{base_name}_ARI.csv"
            metrics_path = folder / f"{base_name}.csv"

            if not ari_path.exists():
                print(f"Missing ARI file: {ari_path}")
                continue

            if not metrics_path.exists():
                print(f"Missing metrics file: {metrics_path}")
                continue

            # Keep row labels like "ARI", "DBCV", etc. as the DataFrame index.
            ari_df = pd.read_csv(ari_path, index_col=0)
            metrics_df = pd.read_csv(metrics_path, index_col=0)

            # ARI file usually has only one row, but this keeps it robust.
            ari_rows = ari_df.loc[["ARI"]] if "ARI" in ari_df.index else ari_df

            # Select only the CVI rows that exist in the file.
            selected_cvis = [cvi for cvi in cvis if cvi in metrics_df.index]
            cvi_rows = metrics_df.loc[selected_cvis]

            # Combine ARI + selected CVIs into one CSV.
            combined = pd.concat([ari_rows, cvi_rows], axis=0)

            # --- SORTING LOGIC ---
            # Check if "ARI" exists in the combined index to avoid errors
            if "ARI" in combined.index:
                # Sort the columns based on the values in the "ARI" row.
                # By default, ascending=True. Change to ascending=False if you want the highest ARI first.
                sorted_columns = combined.loc["ARI"].sort_values(ascending=True).index
                combined = combined[sorted_columns]
            # ---------------------

            out_path = f"./results/test/{base_name}_ARI_plus_CVIs.csv"
            combined.to_csv(out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    combine_ari_and_cvis(
        datasets=["coil20", "olivetti", "yaleA"],
        clusterers=["AgglomerativeClustering", "HDBSCAN", "KMeans", "SpectralClustering"],
        cvis=["DBCV", "AD-S", "AD-idea"],
    )

    combine_ari_and_cvis(
        datasets=["ecoli", "glass", "ionosphere", "sonar", "statlog", "wdbc", "wine", "yeast"],
        clusterers=["AgglomerativeClustering", "DBSCAN", "HDBSCAN", "KMeans", "MeanShift", "SpectralClustering"],
        cvis=["DBCV", "AD-S", "AD-idea"],
    )