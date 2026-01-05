def split_data_and_labels(input_path, data_output_path, labels_output_path):
    with open(input_path, 'r') as fin, \
         open(data_output_path, 'w') as fdata, \
         open(labels_output_path, 'w') as flabel:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split()          # split on whitespace
            label = parts[-1]             # last column
            data = parts[:-1]             # all other columns

            fdata.write(' '.join(data) + '\n')
            flabel.write(label + '\n')


if __name__ == '__main__':
    split_data_and_labels(
        'seeds_dataset.txt',
        'seeds.data',
        'seeds.labels0'
    )
