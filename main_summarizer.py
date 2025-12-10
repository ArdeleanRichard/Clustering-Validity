import pandas as pd
import glob

# FOLDER containing your CSVs
FOLDER = "./results/analysis/*.csv"

# dictionary to accumulate results
metric_correct_counts = {}   # +1 if entire row has ZERO *
metric_error_counts = {}     # +1 per * in any cell of that row

# iterate through all CSV files
for file in glob.glob(FOLDER):

    df = pd.read_csv(file, sep=",")  # change sep if needed
    # ensure metric names are index
    df = df.set_index(df.columns[0])

    for metric in df.index:
        row = df.loc[metric]

        # count * occurrences
        num_errors = row.astype(str).str.contains(r"\*").sum()



        # initialize counters if first time seeing this metric
        metric_correct_counts.setdefault(metric, 0)
        metric_error_counts.setdefault(metric, 0)

        # whole row correct?
        if num_errors == 0:
            metric_correct_counts[metric] += 1
        else:
            if metric == "DBCV (↑)" or metric == "idea (↑)" or metric == "CDbw (↑)":
                print(metric, file)

        # accumulate total * count
        metric_error_counts[metric] += num_errors

# build final summary
summary = pd.DataFrame({
    f"Correct evaluations (out of {len(glob.glob(FOLDER))})": pd.Series(metric_correct_counts),
    f"Errors (out of {5 * len(glob.glob(FOLDER))})": pd.Series(metric_error_counts)
})

print(summary)
