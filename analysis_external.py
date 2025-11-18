import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from analysis_measures import measure_function
from constants import LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS, FOLDER_FIGS_ANALYSIS_EXTERNAL, MAP_EXTERNAL_METRICS
from load_datasets import load_UNBALANCE_STATS, generate_unbalance_like
from ours.external_scores import balanced_external


def analyze_external_metric_vs_imbalance(cvi_str, scales=np.linspace(1.0, 10.0, 10), save_prefix="analysis_imbalance"):
    cvi_name_acronym, cvi_name_full, cvi_function = MAP_EXTERNAL_METRICS[cvi_str]

    UNBALANCE_STATS, MAJORITY_IDS, MINORITY_IDS = load_UNBALANCE_STATS()

    imbalance_vals = []
    scores = []
    balanced_scores = []

    for i, s in enumerate(scales):
        X, y_true = generate_unbalance_like(UNBALANCE_STATS, MAJORITY_IDS, MINORITY_IDS, scale_minority=s)
        ir = measure_function(y_true)

        # Randomize labels ONLY for originally small clusters
        y_rand = y_true.copy()
        for cid in MINORITY_IDS:
            mask = y_true == cid
            if mask.any():
                y_rand[mask] = np.random.choice(MINORITY_IDS, size=mask.sum())

        score = cvi_function(y_true, y_rand)
        balanced_score = balanced_external(cvi_function, y_true, y_rand, method='macro')

        imbalance_vals.append(ir)
        scores.append(score)
        balanced_scores.append(balanced_score)

        # Save scatter plots for verification
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        label_color = [LABEL_COLOR_MAP[i] for i in y_true]
        axes[0].scatter(X[:, 0], X[:, 1], c=label_color, s=10)
        axes[0].set_title(f'True labels (scale={s:.2f}, IR={ir:.2f})')
        label_color = [LABEL_COLOR_MAP[i] for i in y_rand]
        axes[1].scatter(X[:, 0], X[:, 1], c=label_color, s=10)
        axes[1].set_title(f'Randomized minority clusters labels\n{cvi_name_acronym}={score:.3f}, B{cvi_name_acronym}={balanced_score:.3f}')
        plt.tight_layout()
        fig.savefig(f"{FOLDER_FIGS_ANALYSIS_EXTERNAL}/{save_prefix}_{cvi_name_acronym}_step_{i:02d}.png", dpi=200)
        plt.close(fig)

        print(f"Step {i+1}/{len(scales)} | scale={s:.2f} | IR={ir:.2f} | ARI={score:.3f} | BARI={balanced_score:.3f}")

    # Plot ARI vs imbalance ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(imbalance_vals, scores, 'o-', linewidth=2, label='ARI', color='r')
    ax.plot(imbalance_vals, balanced_scores, 'o-', linewidth=2, label='BARI', color='g')
    ax.set_xlabel('Imbalance Ratio (majority/minority)')
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_title('Metric vs Imbalance Ratio\n(on randomized minority clusters)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.legend()
    fig.savefig(f"{FOLDER_FIGS_ANALYSIS_EXTERNAL}/{save_prefix}_{cvi_name_acronym}.png", dpi=200)
    plt.close(fig)

    return imbalance_vals, scores, balanced_scores


if __name__ == "__main__":
    scales = np.linspace(1.0, 10.0, 10)
    analyze_external_metric_vs_imbalance("ari", scales=scales)
    analyze_external_metric_vs_imbalance("ami", scales=scales)
