import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

from scipy.stats import wilcoxon

from constants_maps import MAP_LOWER_IS_BETTER


csv_path = "./results/correlation/saved/realdata_correlations_cvi_to_ari.csv"
df = pd.read_csv(csv_path, index_col=0)


df_processed = df.copy()
for cvi in df_processed.index:
    if cvi.lower() in MAP_LOWER_IS_BETTER:
        df_processed.loc[cvi] = -df_processed.loc[cvi]

# Convert to numeric and handle nulls
df_processed = df_processed.apply(pd.to_numeric, errors='coerce')

# Remove rows with all null values
df_processed = df_processed.dropna(how='all')

# Prepare data for analysis
df_complete = df_processed.dropna()


print(f"\n{'DATASET INFORMATION':-^100}")
print(f"Number of CVIs: {len(df_processed)}")
print(f"Number of datasets: {len(df_complete.columns)}")
print(f" - Dataset names: {', '.join(df_complete.columns.tolist())}")

# ==================================================================================
# PART 1: FRIEDMAN TEST
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 1: FRIEDMAN TEST - OMNIBUS TEST FOR DIFFERENCES AMONG CVIs':-^100}")
print(f"{'=' * 100}")

print("\nTest assumptions and rationale:")
print("  - Non-parametric test (no normality assumption required)")
print("  - Repeated measures design (same datasets used for all CVIs)")
print("  - Tests if CVIs have different distributions of correlations across datasets")
print("  - Null hypothesis: All CVIs perform equivalently across datasets")

if len(df_complete) >= 3:
    data_for_friedman = df_complete.T.values
    statistic, p_value = stats.friedmanchisquare(*[data_for_friedman[:, i] for i in range(data_for_friedman.shape[1])])

    print(f"\nFriedman Test Results:")
    print(f"  Chi-square statistic: {statistic:.4f}")
    print(f"  Degrees of freedom: {len(df_complete) - 1}")
    print(f"  P-value: {p_value:.8f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"  *** SIGNIFICANT at α={alpha}: Reject null hypothesis ***")
        print(f"  Interpretation: CVIs show significantly different correlation patterns with ARI")
    else:
        print(f"  NOT SIGNIFICANT at α={alpha}: Cannot reject null hypothesis")
        print(f"  Interpretation: No evidence of differences among CVIs")

# ==================================================================================
# PART 2: RANKING ANALYSIS
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 2: RANKING ANALYSIS - MEAN RANK ACROSS DATASETS':-^100}")
print(f"{'=' * 100}")

print("\nRanking methodology:")
print("  - For each dataset, CVIs are ranked by correlation (highest = rank 1)")
print("  - Mean rank computed across all datasets")
print("  - Lower mean rank indicates better overall performance")
print("  - Robust to outliers and dataset-specific effects")

n_datasets = len(df_complete.columns)
n_cvis = len(df_complete)

# Calculate ranks for each dataset (1 = best, n = worst)
ranks_per_dataset = df_complete.T.rank(axis=1, ascending=False)
mean_ranks = ranks_per_dataset.mean(axis=0).sort_values()

print(f"\n{'CVIs BY MEAN RANK':-^100}")
print(f"{'Rank':<6} {'CVI':<15} {'Mean Rank':<12} {'Std Rank':<12} {'Best Count':<12} {'Worst Count'}")
print("-" * 100)

rank_std = ranks_per_dataset.std(axis=0)
best_count = (ranks_per_dataset == 1).sum(axis=0)
worst_count = (ranks_per_dataset == n_cvis).sum(axis=0)

for i, (cvi, mean_rank) in enumerate(mean_ranks.items(), 1):
    print(f"{i:<6} {cvi:<15} {mean_rank:<12.2f} {rank_std[cvi]:<12.2f} {best_count[cvi]:<12} {worst_count[cvi]}")

# ==================================================================================
# PART 3: POST-HOC NEMENYI TEST
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 3: POST-HOC NEMENYI TEST - PAIRWISE COMPARISONS':-^100}")
print(f"{'=' * 100}")

print("\nNemenyi test methodology:")
print("  - Post-hoc test following significant Friedman test")
print("  - Controls family-wise error rate across all pairwise comparisons")
print("  - Based on rank differences between CVIs")
print("  - Conservative test (reduces Type I errors)")

# Critical difference for Nemenyi test
# Using Tukey's HSD critical value approximation for Nemenyi
q_alpha_05 = 2.850  # Approximate for large number of groups at α=0.05
q_alpha_01 = 3.314  # Approximate for large number of groups at α=0.01

cd_05 = q_alpha_05 * np.sqrt((n_cvis * (n_cvis + 1)) / (6 * n_datasets))
cd_01 = q_alpha_01 * np.sqrt((n_cvis * (n_cvis + 1)) / (6 * n_datasets))

print(f"\nCritical Differences:")
print(f"  CD (α=0.05): {cd_05:.4f}")
print(f"  CD (α=0.01): {cd_01:.4f}")
print(f"  Rank differences > CD indicate significant performance differences")

# Calculate all pairwise rank differences
pairwise_results = []
for cvi1, cvi2 in combinations(df_complete.index, 2):
    rank_diff = abs(mean_ranks[cvi1] - mean_ranks[cvi2])
    significant_05 = rank_diff > cd_05
    significant_01 = rank_diff > cd_01

    pairwise_results.append({
        'CVI1': cvi1,
        'CVI2': cvi2,
        'Rank_Diff': rank_diff,
        'Sig_0.05': significant_05,
        'Sig_0.01': significant_01
    })

pairwise_df = pd.DataFrame(pairwise_results)
sig_05 = pairwise_df[pairwise_df['Sig_0.05'] == True].sort_values('Rank_Diff', ascending=False)
sig_01 = pairwise_df[pairwise_df['Sig_0.01'] == True].sort_values('Rank_Diff', ascending=False)

print(f"\n{'SIGNIFICANT PAIRWISE DIFFERENCES (α=0.05)':-^100}")
print(f"Total significant pairs at α=0.05: {len(sig_05)}/{len(pairwise_df)}")
if len(sig_05) > 0:
    print(f"\nTop 20 most significant differences:")
    print(f"{'CVI 1':<15} {'CVI 2':<15} {'Rank Diff':<12} {'Significance'}")
    print("-" * 100)
    for idx, row in sig_05.head(20).iterrows():
        sig_level = "***" if row['Sig_0.01'] else "**"
        print(f"{row['CVI1']:<15} {row['CVI2']:<15} {row['Rank_Diff']:<12.2f} {sig_level}")
else:
    print("  No significant pairwise differences found at α=0.05")

print(f"\n{'SIGNIFICANT PAIRWISE DIFFERENCES (α=0.01)':-^100}")
print(f"Total significant pairs at α=0.01: {len(sig_01)}/{len(pairwise_df)}")

# ==================================================================================
# PART 4: CORRELATION MAGNITUDE ANALYSIS
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 4: CORRELATION MAGNITUDE ANALYSIS - ABSOLUTE PERFORMANCE':-^100}")
print(f"{'=' * 100}")

print("\nMethodology:")
print("  - Analyzes actual correlation values (not just ranks)")
print("  - Mean correlation indicates average agreement with ARI")
print("  - Positive correlations = CVI agrees with ground truth")
print("  - Negative correlations = CVI inversely related to ground truth")

mean_corr = df_complete.mean(axis=1).sort_values(ascending=False)
median_corr = df_complete.median(axis=1)
min_corr = df_complete.min(axis=1)
max_corr = df_complete.max(axis=1)
range_corr = max_corr - min_corr

print(f"\n{'CVIs BY MEAN CORRELATION':-^100}")
print(f"{'Rank':<6} {'CVI':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Range'}")
print("-" * 100)

for i, cvi in enumerate(mean_corr.index, 1):
    print(f"{i:<6} {cvi:<15} {mean_corr[cvi]:<10.3f} {median_corr[cvi]:<10.3f} "
          f"{min_corr[cvi]:<10.3f} {max_corr[cvi]:<10.3f} {range_corr[cvi]:.3f}")


# ==================================================================================
# PART 5: ROBUSTNESS ANALYSIS
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 5: ROBUSTNESS ANALYSIS - PERFORMANCE STABILITY':-^100}")
print(f"{'=' * 100}")

print("\nRobustness metrics:")
print("  - Range: Difference between best and worst dataset performance")
print("  - Sign consistency: Proportion of datasets with same-sign correlation")
print("  - Positive agreement: Proportion of datasets with positive correlation")
print("  - Low range + high positive agreement = robust CVI")

positive_count = (df_complete > 0).sum(axis=1)
negative_count = (df_complete < 0).sum(axis=1)
sign_consistency = np.maximum(positive_count, negative_count) / n_datasets
positive_agreement = positive_count / n_datasets

robustness_df = pd.DataFrame({
    'CVI': df_complete.index,
    'Mean_Corr': mean_corr,
    'Range': range_corr,
    'Positive_Agree': positive_agreement,
    'Sign_Consistency': sign_consistency,
    'Mean_Rank': mean_ranks
})

# Most robust: high positive agreement, low range
robust_score = robustness_df['Positive_Agree'] - (robustness_df['Range'] / robustness_df['Range'].max())
robustness_df['Robust_Score'] = robust_score
robustness_sorted = robustness_df.sort_values('Robust_Score', ascending=False)

print(f"\n{'MOST ROBUST CVIs (High Positive Agreement + Low Range)':-^100}")
print(f"{'Rank':<6} {'CVI':<15} {'Mean Corr':<12} {'Pos. Agree':<12} {'Range':<12} {'Robust Score'}")
print("-" * 100)

for i, row in robustness_sorted.head(15).iterrows():
    print(f"{robustness_sorted.index.get_loc(i) + 1:<6} {row['CVI']:<15} {row['Mean_Corr']:<12.3f} "
          f"{row['Positive_Agree']:<12.1%} {row['Range']:<12.3f} {row['Robust_Score']:.3f}")

print(f"\n{'LEAST ROBUST CVIs (Low Positive Agreement or High Range)':-^100}")
print(f"{'Rank':<6} {'CVI':<15} {'Mean Corr':<12} {'Pos. Agree':<12} {'Range':<12} {'Robust Score'}")
print("-" * 100)

for i, row in robustness_sorted.tail(15).iterrows():
    rank = len(robustness_sorted) - robustness_sorted.index.get_loc(i)
    print(f"{rank:<6} {row['CVI']:<15} {row['Mean_Corr']:<12.3f} "
          f"{row['Positive_Agree']:<12.1%} {row['Range']:<12.3f} {row['Robust_Score']:.3f}")

# ==================================================================================
# PART 6: WILCOXON SIGNED-RANK TESTS (PAIRWISE SUPERIORITY)
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 6: WILCOXON SIGNED-RANK TESTS - HEAD-TO-HEAD COMPARISONS':-^100}")
print(f"{'=' * 100}")

print("\nWilcoxon test methodology:")
print("  - Non-parametric paired test comparing two CVIs across datasets")
print("  - Tests if one CVI consistently outperforms another")
print("  - Bonferroni correction applied for multiple comparisons")
print("  - More powerful than Nemenyi for specific comparisons")

# Compare top 5 CVIs with each other
top_5_cvis = mean_corr.head(5).index.tolist()
bottom_5_cvis = mean_corr.tail(5).index.tolist()

print(f"\nTop 5 CVIs: {', '.join(top_5_cvis)}")
print(f"\nPairwise comparisons among top 5 CVIs:")

n_comparisons = len(list(combinations(top_5_cvis, 2)))
bonferroni_alpha = 0.05 / n_comparisons

print(f"Bonferroni-corrected α: {bonferroni_alpha:.6f} (for {n_comparisons} comparisons)")
print(f"\n{'CVI 1':<15} {'CVI 2':<15} {'Statistic':<12} {'P-value':<12} {'Significant':<15} {'Winner'}")
print("-" * 100)

for cvi1, cvi2 in combinations(top_5_cvis, 2):
    data1 = df_complete.loc[cvi1].values
    data2 = df_complete.loc[cvi2].values

    try:
        stat, p_val = wilcoxon(data1, data2, alternative='two-sided')
        is_sig = "***" if p_val < bonferroni_alpha else "ns"
        winner = cvi1 if data1.mean() > data2.mean() else cvi2
        print(f"{cvi1:<15} {cvi2:<15} {stat:<12.2f} {p_val:<12.6f} {is_sig:<15} {winner}")
    except:
        print(f"{cvi1:<15} {cvi2:<15} {'N/A':<12} {'N/A':<12} {'Error':<15} {'N/A'}")

# Compare best CVI with worst CVIs
best_cvi = mean_corr.index[0]
print(f"\n{'Best CVI vs Bottom 5 CVIs':-^100}")
print(f"Best CVI: {best_cvi} (Mean Correlation: {mean_corr[best_cvi]:.3f})")
print(f"\n{'Best CVI':<15} {'Compared to':<15} {'Statistic':<12} {'P-value':<12} {'Significant':<15} {'Effect Size'}")
print("-" * 100)

n_comparisons_best = len(bottom_5_cvis)
bonferroni_alpha_best = 0.05 / n_comparisons_best

for worst_cvi in bottom_5_cvis:
    data_best = df_complete.loc[best_cvi].values
    data_worst = df_complete.loc[worst_cvi].values

    try:
        stat, p_val = wilcoxon(data_best, data_worst, alternative='greater')
        is_sig = "***" if p_val < bonferroni_alpha_best else "ns"
        effect_size = (data_best.mean() - data_worst.mean()) / np.std(data_best - data_worst)
        print(f"{best_cvi:<15} {worst_cvi:<15} {stat:<12.2f} {p_val:<12.6f} {is_sig:<15} {effect_size:.3f}")
    except:
        print(f"{best_cvi:<15} {worst_cvi:<15} {'N/A':<12} {'N/A':<12} {'Error':<15} {'N/A'}")

# ==================================================================================
# PART 7: DATASET-SPECIFIC ANALYSIS
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 7: DATASET-SPECIFIC ANALYSIS - PERFORMANCE BY DATASET':-^100}")
print(f"{'=' * 100}")

print("\nDataset characteristics:")
print("  - Best CVI: Highest correlation for that dataset")
print("  - Worst CVI: Lowest correlation for that dataset")
print("  - Range: Difference between best and worst CVI")
print("  - Shows which datasets have clearer CVI performance differentiation")

print(f"\n{'Dataset':<15} {'Best CVI':<15} {'Best Corr':<12} {'Worst CVI':<15} {'Worst Corr':<12} {'Range'}")
print("-" * 100)

for dataset in df_complete.columns:
    best_idx = df_complete[dataset].idxmax()
    worst_idx = df_complete[dataset].idxmin()
    best_val = df_complete[dataset].max()
    worst_val = df_complete[dataset].min()
    dataset_range = best_val - worst_val

    print(f"{dataset:<15} {best_idx:<15} {best_val:<12.3f} {worst_idx:<15} {worst_val:<12.3f} {dataset_range:.3f}")

# ==================================================================================
# PART 8: FINAL RECOMMENDATIONS
# ==================================================================================
print(f"\n{'=' * 100}")
print(f"{'PART 8: SUMMARY AND RECOMMENDATIONS':-^100}")
print(f"{'=' * 100}")

print("\nTop 3 Recommended CVIs (Based on Multiple Criteria):")

# Combined scoring
recommendation_score = (
        (mean_corr - mean_corr.min()) / (mean_corr.max() - mean_corr.min()) * 0.4 +  # 40% weight on mean correlation
        (1 - (mean_ranks - mean_ranks.min()) / (mean_ranks.max() - mean_ranks.min())) * 0.3 +  # 30% weight on rank
        robustness_df.set_index('CVI')['Robust_Score'].rank(pct=True) * 0.3  # 30% weight on robustness
)

top_3_recommended = recommendation_score.sort_values(ascending=False).head(3)

for i, (cvi, score) in enumerate(top_3_recommended.items(), 1):
    print(f"\n{i}. {cvi}")
    print(f"   - Mean Correlation: {mean_corr[cvi]:.3f}")
    print(f"   - Mean Rank: {mean_ranks[cvi]:.2f}")
    print(f"   - Positive Agreement: {positive_agreement[cvi]:.1%}")
    print(f"   - Range: {range_corr[cvi]:.3f}")
    print(f"   - Overall Score: {score:.3f}")
