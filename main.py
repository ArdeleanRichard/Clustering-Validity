import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from constants import LABEL_COLOR_MAP, FOLDER_RESULTS, FOLDER_FIGS_DATA
from load_datasets import create_set1, create_set_g, create_set_a, create_set_s
from load_labelsets import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line
from load_metrics import choose_metric, create_metric_table, create_metric_table_with_arrows


def remove_dups(X, gt):
    uniq_rows, idx = np.unique(X, axis=0, return_index=True)
    keep_idx = np.sort(idx)  # sort to preserve original order of first occurrences
    X = X[keep_idx]
    gt = gt[keep_idx]
    return X, gt

def reencode(labels):
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    return encoded

def run_score_set(datasets, plot=False):
    for name_data, (X, gt) in datasets:
        # X, gt = shuffle(X, gt, random_state=random_state)
        scale = (-1, 1)
        midpoint = np.mean(scale)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)
        X = MinMaxScaler(scale).fit_transform(X)

        # Generate label sets
        dfl = assign_labels_by_given_line(X, diagonal_line(X, "first"))
        dsl = assign_labels_by_given_line(X, diagonal_line(X, "first"))
        dp = assign_labels_by_given_line(X, diagonal_line(X, "first"))
        vl = assign_labels_by_given_line(X, vertical_line(midpoint))
        hl = assign_labels_by_given_line(X, horizontal_line(midpoint))
        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        label_sets = {"gt": gt, "dfl": dfl, "dsl": dsl, "vl": vl, "hl": hl, "rl": rl}

        # Create and print metric table
        create_metric_table_with_arrows(X, label_sets=label_sets, save=f"{FOLDER_RESULTS}/metrics_{name_data}.csv", prnt=True)

        if plot:
            for name_labelset, labels in zip(["gt", "dfl", "dsl", "vl", "hl", "rl"], [gt, dp, vl, hl, rl]):
                if len(np.unique(labels)) > len(LABEL_COLOR_MAP.keys()):
                    cmap = plt.cm.get_cmap("gist_ncar", len(labels))
                    label_color = [cmap(i) for i in range(len(labels))]
                else:
                    # use your custom colors
                    label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"{FOLDER_FIGS_DATA}/svgs/{name_data}_{name_labelset}.svg")
                plt.savefig(f"{FOLDER_FIGS_DATA}/{name_data}_{name_labelset}.png")
                plt.close()



def run_scores_set1(plot=False):
    # datasets = create_set1(n_samples=1000)
    # datasets = create_set_g(dims=2)
    # datasets = create_set_a()
    datasets = create_set_s()

    run_score_set(datasets, plot)




if __name__ == '__main__':
    random_state = 42

    run_scores_set1(plot=True)
