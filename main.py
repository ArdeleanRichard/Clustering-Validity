from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from constants import FOLDER_FIGS_DATA, scale, FOLDER_RESULTS_CVIS
from constants_maps import MAP_LABELSET_TO_NAME
from load_datasets import create_set1, create_set_g, create_set_a, create_set_s, create_set_graves, create_set_sipu, create_set_uci, create_set_wut
from load_CVIs import create_indices_table_with_arrows
from utils import remove_dups, reencode, load_labelsets, choose_colors


def run_score_set(datasets, list_labelsets=["dfl", "dsl", "vl", "hl", "rl"], plot=False):
    for name_data, (X, gt) in datasets:
        print(name_data)
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



def run_scores(plot=False):
    ## Each element in the list is a set of datasets
    sets = [
        create_set1(n_samples=1000),
        create_set_a(),
        create_set_s(),
        create_set_graves(),
        create_set_sipu(),
        create_set_wut(),
    ]
    for set in sets:
        run_score_set(set, plot=plot)



if __name__ == '__main__':
    run_scores(plot=True)
