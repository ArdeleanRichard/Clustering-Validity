import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from typing import Dict, Tuple, Optional, List

from constants import LABEL_COLOR_MAP, FOLDER_FIGS_ANALYSIS
# from load_datasets import create_unbalance
#
# def cluster_stats(X, labels):
#     X = np.asarray(X)
#     labels = np.asarray(labels)
#     stats = {}
#
#     for label in np.unique(labels):
#         cluster_points = X[labels == label]
#         n = len(cluster_points)
#         mean = cluster_points.mean(axis=0)
#         cov = np.cov(cluster_points, rowvar=False)
#
#         stats[int(label)] = {
#             "n": int(n),
#             "mean": mean.tolist(),
#             "cov": cov.tolist()
#         }
#
#     return stats
#
# X, labels = create_unbalance()
# stats = cluster_stats(X, labels)
#
# for k, v in stats.items():
#     print(f"    {k}: {v},")


UNBALANCE_STATS: Dict[int, Dict] = {
    0: {'n': 2000, 'mean': [150006.7365, 350103.876], 'cov': [[9982716.372253874, 1193299.6986753377], [1193299.6986753377, 10042633.918583289]]},
    1: {'n': 2000, 'mean': [179954.98, 380007.9705], 'cov': [[9869979.581390698, 1179290.3195697847], [1179290.3195697847, 9674416.364812154]]},
    2: {'n': 2000, 'mean': [209948.245, 349963.26], 'cov': [[9665368.451200599, 517801.0848424211], [517801.0848424211, 9168240.486643318]]},
    3: {'n': 100, 'mean': [440754.33, 298283.2], 'cov': [[85839933.63747482, -2361146.793939391], [-2361146.793939391, 89169344.6868687]]},
    4: {'n': 100, 'mean': [440134.41, 400135.41], 'cov': [[120445372.18373743, 4221838.76959596], [4221838.76959596, 125848151.05242425]]},
    5: {'n': 100, 'mean': [491036.01, 349798.33], 'cov': [[107976910.91909094, -3136604.417474747], [-3136604.417474747, 97399620.34454548]]},
    6: {'n': 100, 'mean': [539379.19, 299652.83], 'cov': [[97408312.98373738, 1546637.083131312], [1546637.083131312, 94213029.71828282]]},
    7: {'n': 100, 'mean': [538883.52, 400947.36], 'cov': [[93971552.77737373, 6441541.154343435], [6441541.154343435, 75038436.87919189]]},
}

# Identify minority clusters (5 smallest)
MINORITY_IDS = sorted(UNBALANCE_STATS.keys(), key=lambda k: UNBALANCE_STATS[k]['n'])[:5]
MAJORITY_IDS = [k for k in UNBALANCE_STATS.keys() if k not in MINORITY_IDS]


def generate_unbalance_like(
    scale_minority: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data similar to unbalance.csv with option to scale minority clusters."""
    if rng is None:
        rng = np.random.default_rng()
    X_parts, y_parts = [], []
    for cid, stats in UNBALANCE_STATS.items():
        n = stats['n']
        if cid in MINORITY_IDS:
            n = int(n * scale_minority)
        mean = np.array(stats['mean'])
        cov = np.array(stats['cov'])
        Xi = rng.multivariate_normal(mean, cov, size=n)
        yi = np.full(n, cid)
        X_parts.append(Xi)
        y_parts.append(yi)
    return np.vstack(X_parts), np.concatenate(y_parts)


def imbalance_ratio(labels: np.ndarray) -> float:
    counts = np.bincount(labels)
    nonzero = counts[counts > 0]
    return nonzero.max() / nonzero.min()

def analyze_ari_vs_imbalance(
    scales: np.ndarray = np.linspace(1.0, 10.0, 10),
    rng: Optional[np.random.Generator] = None,
    save_prefix: str = "analysis_imbalance_ari"
) -> Tuple[List[float], List[float]]:
    if rng is None:
        rng = np.random.default_rng(42)

    imbalance_vals = []
    ari_vals = []

    for i, s in enumerate(scales):
        X, y_true = generate_unbalance_like(scale_minority=s, rng=rng)
        ir = imbalance_ratio(y_true)

        # Randomize labels ONLY for originally small clusters
        y_rand = y_true.copy()
        for cid in MINORITY_IDS:
            mask = y_true == cid
            if mask.any():
                y_rand[mask] = rng.choice(MINORITY_IDS, size=mask.sum())

        ari = adjusted_rand_score(y_true, y_rand)

        imbalance_vals.append(ir)
        ari_vals.append(ari)

        # Save scatter plots for verification
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        label_color = [LABEL_COLOR_MAP[i] for i in y_true]
        axes[0].scatter(X[:, 0], X[:, 1], c=label_color, s=10)
        axes[0].set_title(f'True labels (scale={s:.2f}, IR={ir:.2f})')
        label_color = [LABEL_COLOR_MAP[i] for i in y_rand]
        axes[1].scatter(X[:, 0], X[:, 1], c=label_color, s=10)
        axes[1].set_title(f'Randomized small labels (ARI={ari:.3f})')
        plt.tight_layout()
        fig.savefig(f"{FOLDER_FIGS_ANALYSIS}/{save_prefix}_step_{i:02d}.png", dpi=200)
        plt.close(fig)

        print(f"Step {i+1}/{len(scales)} | scale={s:.2f} | IR={ir:.2f} | ARI={ari:.3f}")

    # Plot ARI vs imbalance ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(imbalance_vals, ari_vals, 'o-', linewidth=2)
    ax.set_xlabel('Imbalance Ratio (majority/minority)')
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_title('ARI vs Imbalance Ratio (Randomized Small Clusters)')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.savefig(f"{FOLDER_FIGS_ANALYSIS}/{save_prefix}.png", dpi=200)
    plt.close(fig)

    return imbalance_vals, ari_vals


def main():
    rng = np.random.default_rng(42)
    scales = np.linspace(1.0, 10.0, 10)
    analyze_ari_vs_imbalance(scales=scales, rng=rng)


if __name__ == "__main__":
    main()
