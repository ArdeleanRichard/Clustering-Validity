from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from constants import FOLDER_FIGS_DATA, scale, FOLDER_RESULTS_CVIS
from constants_maps import MAP_LABELSET_TO_NAME, LIST_LABELSETS
from load_CVIs import create_indices_table_with_arrows
from utils import remove_dups, reencode, load_labelsets, choose_colors



def main_synthetic_data_with_labelsets(plot=False):
    from load_datasets import create_synthetic_datasets
    datasets = create_synthetic_datasets()

    for data_name, (X, gt) in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {data_name} {X.shape}")
        print(f"{'=' * 80}")
        # X, gt = shuffle(X, gt, random_state=random_state)
        X, gt = remove_dups(X, gt)
        gt = reencode(gt)
        X = MinMaxScaler(scale).fit_transform(X)

        label_sets = {"gt": gt}
        label_sets = load_labelsets(X, gt, scale, label_sets, list_labelsets=LIST_LABELSETS)

        # Create and print metric table
        create_indices_table_with_arrows(X, label_sets=label_sets, save=f"{FOLDER_RESULTS_CVIS}/CVIs_{data_name}.csv", prnt=True)

        if plot:
            for name_labelset, labels in label_sets.items():
                label_color = choose_colors(labels)

                plt.title(MAP_LABELSET_TO_NAME[name_labelset], fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=14)
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"{FOLDER_FIGS_DATA}/svgs/{data_name}_{name_labelset}.svg")
                plt.savefig(f"{FOLDER_FIGS_DATA}/{data_name}_{name_labelset}.png")
                plt.close()

def main_summarize():
    import pandas as pd
    import glob
    from constants import FOLDER_RESULTS_CVIS
    FOLDER = FOLDER_RESULTS_CVIS + "/*.csv"

    metric_correct_counts = {}  # +1 if entire row has ZERO *
    metric_error_counts = {}  # +1 for each * in any cell of that row

    for file in glob.glob(FOLDER):

        df = pd.read_csv(file, sep=",")

        # ensure metric names are index
        df = df.set_index(df.columns[0])

        for metric in df.index:
            row = df.loc[metric]

            # count * occurrences
            num_errors = row.astype(str).str.contains(r"\*").sum()

            # initialize counters if first time seeing this metric
            metric_correct_counts.setdefault(metric, 0)
            metric_error_counts.setdefault(metric, 0)

            if num_errors == 0:
                # whole row correct
                metric_correct_counts[metric] += 1
            # else:
            #     if metric == "DBCV (↑)" or metric == "AD-idea (↑)" or metric == "CDbw (↑)":
            #         print(metric, file)

            # accumulate total * count
            metric_error_counts[metric] += num_errors

    summary = pd.DataFrame({
        f"Correct evaluations (out of {len(glob.glob(FOLDER))})": pd.Series(metric_correct_counts),
        f"Errors (out of {5 * len(glob.glob(FOLDER))})": pd.Series(metric_error_counts)
    })

    summary.to_csv(FOLDER_RESULTS_CVIS + "/.summary.csv")
    print(summary)

if __name__ == '__main__':
    # main_synthetic_data_with_labelsets(plot=False)
    main_summarize()
