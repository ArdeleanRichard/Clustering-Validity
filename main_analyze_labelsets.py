from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from constants import FOLDER_FIGS_DATA, scale, FOLDER_RESULTS_CVIS
from constants_maps import MAP_LABELSET_TO_NAME
from load_CVIs import create_indices_table_with_arrows
from utils import remove_dups, reencode, load_labelsets, choose_colors


def run_CVIs(datasets, list_labelsets=["dfl", "dsl", "vl", "hl", "rl"], plot=False):
    for name_data, (X, gt) in datasets:
        print(name_data, len(X))
        # X, gt = shuffle(X, gt, random_state=random_state)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)
        X = MinMaxScaler(scale).fit_transform(X)

        label_sets = {"gt": gt}
        label_sets = load_labelsets(X, gt, scale, label_sets, list_labelsets=list_labelsets)

        # Create and print metric table
        create_indices_table_with_arrows(X, label_sets=label_sets, save=f"{FOLDER_RESULTS_CVIS}/metrics_{name_data}.csv", prnt=True)

        if plot:
            for name_labelset, labels in label_sets.items():
                label_color = choose_colors(labels)

                plt.title(MAP_LABELSET_TO_NAME[name_labelset], fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=14)
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"{FOLDER_FIGS_DATA}/svgs/{name_data}_{name_labelset}.svg")
                plt.savefig(f"{FOLDER_FIGS_DATA}/{name_data}_{name_labelset}.png")
                plt.close()



def main_synthetic_data_with_labelsets(plot=False):
    from load_datasets import create_synthetic_datasets
    datasets = create_synthetic_datasets()

    run_CVIs(datasets, plot=plot)


if __name__ == '__main__':
    main_synthetic_data_with_labelsets(plot=False)
