import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score

from cvis_ours.ad_CVIs import ad_calinski_harabasz_score
from cvis_ours.ad_CVIs import ad_idea

# -----------------------------
# Parameters
# -----------------------------
rng = np.random.default_rng(42)

n_clusters = 2
points_per_cluster = 50
n_points = n_clusters * points_per_cluster

noise_levels = np.arange(0, 50, 1)   # 0% to 50%
n_trials = 50                        # repeated tests per noise level

# -----------------------------
# Generate well-separated data
# -----------------------------
mean_1 = np.array([0.0, 0.0])
mean_2 = np.array([8.0, 0.0])
std = 0.7

X1 = rng.normal(mean_1, std, size=(points_per_cluster, 2))
X2 = rng.normal(mean_2, std, size=(points_per_cluster, 2))
X = np.vstack([X1, X2])

y_true = np.array([0]*points_per_cluster + [1]*points_per_cluster)

# -----------------------------
# Containers
# -----------------------------
ari_scores = np.zeros((len(noise_levels), n_trials))
ch_scores = np.zeros((len(noise_levels), n_trials))
s_scores = np.zeros((len(noise_levels), n_trials))
adch_scores = np.zeros((len(noise_levels), n_trials))
adidea_scores = np.zeros((len(noise_levels), n_trials))

# -----------------------------
# Noise experiment
# -----------------------------
for i, pct in enumerate(noise_levels):
    n_flip = int(round(pct / 100 * n_points))

    for t in range(n_trials):
        y_noisy = y_true.copy()

        if n_flip > 0:
            idx = rng.choice(n_points, n_flip, replace=False)
            y_noisy[idx] = 1 - y_noisy[idx]   # flip labels

        ari_scores[i, t] = adjusted_rand_score(y_true, y_noisy)
        s_scores[i, t] = silhouette_score(X, y_noisy)
        ch_scores[i, t] = calinski_harabasz_score(X, y_noisy)
        adch_scores[i, t] = ad_calinski_harabasz_score(X, y_noisy)
        adidea_scores[i, t] = ad_idea(X, y_noisy)

# -----------------------------
# Aggregate statistics
# -----------------------------
ari_mean = ari_scores.mean(axis=1)
ari_std = ari_scores.std(axis=1)

s_scores = s_scores / s_scores.max()
s_mean = s_scores.mean(axis=1)
s_std = s_scores.std(axis=1)

ch_scores = ch_scores / ch_scores.max()
ch_mean = ch_scores.mean(axis=1)
ch_std = ch_scores.std(axis=1)

adch_scores = adch_scores / adch_scores.max()
adch_mean = adch_scores.mean(axis=1)
adch_std = adch_scores.std(axis=1)

adidea_scores = adidea_scores / adidea_scores.max()
adidea_mean = adidea_scores.mean(axis=1)
adidea_std = adidea_scores.std(axis=1)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(noise_levels*2, ari_mean, label="Adjusted Rand Index")
plt.fill_between(
    noise_levels*2,
    ari_mean - ari_std,
    ari_mean + ari_std,
    alpha=0.2
)

plt.plot(
    noise_levels*2,
    ch_mean,
    linestyle="--",
    label="Calinski-Harabasz"
)
plt.fill_between(
    noise_levels*2,
    ch_mean - ch_std,
    ch_mean + ch_std,
    alpha=0.2
)

plt.plot(
    noise_levels*2,
    adch_mean,
    linestyle="--",
    label="AD Calinski-Harabasz"
)
plt.fill_between(
    noise_levels*2,
    adch_mean - adch_std,
    adch_mean + adch_std,
    alpha=0.2
)


plt.plot(
    noise_levels*2,
    adidea_mean,
    linestyle="--",
    label="AD idea"
)
plt.fill_between(
    noise_levels*2,
    adidea_mean - adidea_std,
    adidea_mean + adidea_std,
    alpha=0.2
)

plt.plot(
    noise_levels*2,
    s_mean,
    linestyle="--",
    label="Silhouette"
)
plt.fill_between(
    noise_levels*2,
    s_mean - s_std,
    s_mean + s_std,
    alpha=0.2
)


plt.xlabel("Label noise (% flipped)")
plt.ylabel("Score (−1 to 1)")
plt.title("Effect of Label Noise on ARI and Calinski–Harabasz")
plt.ylim(-1.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


