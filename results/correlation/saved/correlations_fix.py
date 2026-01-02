import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

from scipy.stats import wilcoxon

from constants_maps import MAP_LOWER_IS_BETTER

def main_fix_corr_values(csv_name, new_name):
    df = pd.read_csv(csv_name, index_col=0)

    df_processed = df.copy()
    for cvi in df_processed.index:
        if cvi.lower() in MAP_LOWER_IS_BETTER:
            df_processed.loc[cvi] = -df_processed.loc[cvi]

    df_processed.to_csv(new_name)

if __name__ == "__main__":
    # main_fix_corr_values(csv_name="realdata_correlations_cvi_to_ari.csv",       new_name="realdata_fix_correlations_cvi_to_ari.csv")
    # main_fix_corr_values(csv_name="realdata_correlations_cvi_to_bari.csv",      new_name="realdata_fix_correlations_cvi_to_bari.csv")
    # main_fix_corr_values(csv_name="realdata_correlations_cvi_to_ari_nn.csv",    new_name="realdata_fix_correlations_cvi_to_ari_nn.csv")
    # main_fix_corr_values(csv_name="realdata_correlations_cvi_to_bari_nn.csv",   new_name="realdata_fix_correlations_cvi_to_bari_nn.csv")

    for algo in ['DBSCAN', 'HDBSCAN', 'MeanShift', 'AgglomerativeClustering', 'SpectralClustering', 'KMeans']:
        main_fix_corr_values(csv_name=f"correlations_cvi_to_ari_{algo}.csv",       new_name=f"correlations_fix_cvi_to_ari_{algo}.csv")