import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from constants import LABEL_COLOR_MAP, FOLDER_RESULTS, FOLDER_FIGS_DATA
from load_datasets import create_set1, create_set_g, create_set_a, create_set_s, create_set_graves, create_set_sipu, \
    create_set_uci, create_set_wut
from load_labelsets import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line
from load_indices import choose_index, create_indices_table, create_indices_table_with_arrows


def remove_dups(X, gt):
    uniq_rows, idx = np.unique(X, axis=0, return_index=True)
    keep_idx = np.sort(idx)  # sort to preserve original order of first occurrences
    X = X[keep_idx]
    gt = gt[keep_idx]
    return X, gt

def reencode(labels):
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    return encoded


def load_labelsets(X, gt, scale, label_sets, list_labelsets):
    midpoint = np.mean(scale)

    # Generate label sets
    if "dfl" in list_labelsets:
        dfl = assign_labels_by_given_line(X, diagonal_line(X, "first"))
        label_sets["dfl"] = dfl
    if "dsl" in list_labelsets:
        dsl = assign_labels_by_given_line(X, diagonal_line(X, "second"))
        label_sets["dsl"] = dsl
    if "vl" in list_labelsets:
        vl = assign_labels_by_given_line(X, vertical_line(midpoint))
        label_sets["vl"] = vl
    if "hl" in list_labelsets:
        hl = assign_labels_by_given_line(X, horizontal_line(midpoint))
        label_sets["hl"] = hl
    if "rl" in list_labelsets:
        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))
        label_sets["rl"] = rl

    return label_sets


def choose_colors(labels):
    label_color = [LABEL_COLOR_MAP[i] for i in labels]

    return label_color


def run_score_set(datasets, list_labelsets=["dfl", "dsl", "vl", "hl", "rl"], plot=False):
    for name_data, (X, gt) in datasets:
        # X, gt = shuffle(X, gt, random_state=random_state)
        scale = (-1, 1)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)
        X = MinMaxScaler(scale).fit_transform(X)

        label_sets = {"gt": gt}
        label_sets = load_labelsets(X, gt, scale, label_sets, list_labelsets)

        # Create and print metric table
        create_indices_table_with_arrows(X, label_sets=label_sets, save=f"{FOLDER_RESULTS}/metrics_{name_data}.csv", prnt=True)

        if plot:
            for name_labelset, labels in label_sets.items():
                label_color = choose_colors(labels)
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"{FOLDER_FIGS_DATA}/svgs/{name_data}_{name_labelset}.svg")
                plt.savefig(f"{FOLDER_FIGS_DATA}/{name_data}_{name_labelset}.png")
                plt.close()



def run_scores_set1(plot=False):
    datasets = create_set1(n_samples=1000)
    # datasets = create_set_g(dims=2)
    # datasets = create_set_a()
    # datasets = create_set_s()
    # datasets = create_set_graves()
    # datasets = create_set_sipu()
    # datasets = create_set_wut()

    run_score_set(datasets, plot=plot)

    # datasets = create_set_uci()
    # run_score_set(datasets, list_labelsets=["rl"], plot=False)





if __name__ == '__main__':
    run_scores_set1(plot=True)
