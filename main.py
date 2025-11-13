import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from constants import LABEL_COLOR_MAP, FOLDER_RESULTS, FOLDER_FIGS_DATA
from load_datasets import create_set1
from load_labelsets import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line
from load_metrics import choose_metric, create_metric_table, create_metric_table_with_arrows


def run_score_set1(datasets, plot=False):
    for i_dataset, (X, gt) in enumerate(datasets):
        # X, gt = shuffle(X, gt, random_state=random_state)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        # Generate label sets
        dp = assign_labels_by_given_line(X, diagonal_line(X))
        vl = assign_labels_by_given_line(X, vertical_line(0))
        hl = assign_labels_by_given_line(X, horizontal_line(0))
        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        label_sets = {"gt": gt, "dp": dp, "vl": vl, "hl": hl, "rl": rl}

        # Create and print metric table
        create_metric_table_with_arrows(X, choose_metric, label_sets=label_sets, save=f"{FOLDER_RESULTS}/metrics_data{i_dataset + 1}.csv", printt=True)

        if plot:
            for name, labels in zip(["gt", "dp", "vl", "hl", "rl"], [gt, dp, vl, hl, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"{FOLDER_FIGS_DATA}/data{i_dataset + 1}_{name}.svg")
                plt.savefig(f"{FOLDER_FIGS_DATA}/data{i_dataset + 1}_{name}.png")
                plt.close()



def run_scores_set1(plot=False):
    n_samples = 1000
    datasets = create_set1(n_samples)

    run_score_set1(datasets, plot)




if __name__ == '__main__':
    random_state = 42

    run_scores_set1(plot=False)
