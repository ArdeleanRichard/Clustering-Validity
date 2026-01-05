import pandas as pd

def split_csv_to_data_and_labels(input_csv: str,
                                 output_data: str,
                                 output_labels: str,
                                 label_col: str) -> None:
    df = pd.read_csv(input_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in input CSV columns: {list(df.columns)}")

    labels_df = df[[label_col]].copy()

    data_df = df.drop(columns=[label_col]).copy()
    labels_df.to_csv(output_labels, index=False, header=False)
    data_df.to_csv(output_data, sep=' ', index=False, header=False)


if __name__ == '__main__':
    split_csv_to_data_and_labels(
        input_csv="sobar-72.csv",
        output_data="ccbr.data",
        output_labels="ccbr.labels0",
        label_col="ca_cervix"
    )

    split_csv_to_data_and_labels(
        input_csv="Wholesale customers data.csv",
        output_data="wholesale.data",
        output_labels="wholesale.labels0",
        label_col="Region"
    )