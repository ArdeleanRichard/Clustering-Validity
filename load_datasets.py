import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

seed = 30
random_state = 170


def create_data1(n_samples):
    avgPoints = n_samples // 3
    C1 = [-5, -10] + .8 * np.random.randn(avgPoints, 2)
    C2 = [5, -10] + .8 * np.random.randn(avgPoints, 2)
    C3 = [5, 10] + .8 * np.random.randn(avgPoints, 2)

    X = np.vstack((C1, C2, C3))

    c1Labels = np.full(len(C1), 0)
    c2Labels = np.full(len(C2), 1)
    c3Labels = np.full(len(C3), 2)

    y = np.hstack((c1Labels, c2Labels, c3Labels))

    data1 = (X, y)

    return data1

def create_data2(n_samples):
    avgPoints = n_samples // 5
    C1 = [5, -10] + .8 * np.random.randn(avgPoints, 2)
    C2 = [0, -9] + .8 * np.random.randn(avgPoints, 2)
    C3 = [-5, -5] + .8 * np.random.randn(avgPoints, 2)
    C4 = [1, 0] + .8 * np.random.randn(avgPoints, 2)
    C5 = [8, -1] + .8 * np.random.randn(avgPoints, 2)

    X = np.vstack((C1, C2, C3, C4, C5))

    c1Labels = np.full(len(C1), 0)
    c2Labels = np.full(len(C2), 1)
    c3Labels = np.full(len(C3), 2)
    c4Labels = np.full(len(C4), 3)
    c5Labels = np.full(len(C5), 4)

    y = np.hstack((c1Labels, c2Labels, c3Labels, c4Labels, c5Labels))

    data2 = (X, y)

    return data2



def create_data3(n_samples):
    return datasets.make_blobs(n_samples=n_samples, random_state=seed)


def create_data4(n_samples):
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=1.0, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    return aniso

def create_data5(n_samples, n_features=2):
    # data5 with data3 variances
    return datasets.make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)


def create_data6(n_samples):
    return datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)


def create_data7(n_samples):
    return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)








def transform_arff_data(data):
    X = []
    y = []
    for sample in data:
        x = []
        for id, value in enumerate(sample):
            if id == len(sample) - 1:
                y.append(value)
            else:
                x.append(value)
        X.append(x)


    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)





def read_uci(fetched_data):
    X = fetched_data.data.features.to_numpy()
    y = fetched_data.data.targets.to_numpy().squeeze()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y


def create_ecoli():
    # data, meta = arff.loadarff('./data/ecoli.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=39)
    return read_uci(fetched_data)

def create_glass():
    # data, meta = arff.loadarff('./data/glass.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=42)
    return read_uci(fetched_data)


def create_yeast():
    # data, meta = arff.loadarff('./data/yeast.arff')
    # return transform_arff_data(data)

    fetched_data = fetch_ucirepo(id=110)
    return read_uci(fetched_data)


def create_statlog():
    fetched_data = fetch_ucirepo(id=147)
    return read_uci(fetched_data)

def create_wdbc():
    fetched_data = fetch_ucirepo(id=17)
    return read_uci(fetched_data)


def create_wine():
    fetched_data = fetch_ucirepo(id=109)
    return read_uci(fetched_data)



def create_unbalance():
    data = pd.read_csv('./data/unbalance.csv', header=None)
    temp_data = data.to_numpy()
    data = (temp_data[:, :-1], temp_data[:, -1])
    return data






def create_2d4c():
    data, meta = arff.loadarff('./data/2d-4c-no4.arff')
    return transform_arff_data(data)

def create_2d10c():
    data, meta = arff.loadarff('./data/2d-10c.arff')
    return transform_arff_data(data)

def create_2d20c():
    data, meta = arff.loadarff('./data/2d-20c-no0.arff')
    return transform_arff_data(data)

def create_3spiral():
    data, meta = arff.loadarff('./data/3-spiral.arff')
    return transform_arff_data(data)

def create_aggregation():
    data, meta = arff.loadarff('./data/aggregation.arff')
    return transform_arff_data(data)

def create_compound():
    data, meta = arff.loadarff('./data/compound.arff')
    return transform_arff_data(data)

def create_elly_2d10c13s():
    data, meta = arff.loadarff('./data/elly-2d10c13s.arff')
    return transform_arff_data(data)





def read_data_and_labels(data_path, labels_path):
    f_data = open(data_path, 'r')
    X = np.array(
        [list(map(float, line.strip().split())) for line in f_data if line.strip()],
        dtype=float
    )
    f_data.close()

    f_labels = open(labels_path, 'r')
    y = np.array(
        [int(line.strip()) for line in f_labels if line.strip()],
        dtype=int
    )
    f_labels.close()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: {X.shape[0]} samples in data but {y.shape[0]} labels.")

    return (X, y)


def create_s(n=1):
    # data, meta = arff.loadarff(f'./data/s/s-set{n}.arff')
    # return transform_arff_data(data)
    return read_data_and_labels(f"./data/s/s{n}.data", f"./data/s/s{n}.labels0")

def create_a(n=1):
    return read_data_and_labels(f"./data/a/a{n}.data", f"./data/a/a{n}.labels0")


def create_g(dims=2, overlap=10):
    return read_data_and_labels(f"./data/g2mg/g2mg_{dims}_{overlap}.data", f"./data/g2mg/g2mg_{dims}_{overlap}.labels0")


def create_ring(type):
    return read_data_and_labels(f"./data/graves/ring{type}.data", f"./data/graves/ring{type}.labels0")

def create_zigzag(type):
    return read_data_and_labels(f"./data/graves/zigzag{type}.data", f"./data/graves/zigzag{type}.labels0")

def create_parabolic():
    return read_data_and_labels(f"./data/graves/parabolic.data", f"./data/graves/parabolic.labels0")

def create_set_graves():
    datasets = []
    datasets.extend([("parabolic", create_parabolic())])
    datasets.extend([(f"ring{t}", create_ring(t)) for t in ["", "_noisy", "_outliers"]])
    datasets.extend([(f"zigzag{t}", create_zigzag(t)) for t in ["", "_noisy", "_outliers"]])

    return datasets


def create_x(n):
    return read_data_and_labels(f"./data/wut/x{n}.data", f"./data/wut/x{n}.labels0")

def create_trajectories():
    return read_data_and_labels(f"./data/wut/trajectories.data", f"./data/wut/trajectories.labels0")

def create_set_wut():
    datasets = []
    datasets.extend([("trajectories", create_trajectories())])
    datasets.extend([(f"x{i}", create_x(i)) for i in [1,2,3]])

    return datasets


def create_set1(n_samples):
    datasets = [
        ("data1", create_data1(n_samples)),
        ("data2", create_data2(n_samples)),
        ("data3", create_data3(n_samples)),
        ("data4", create_data4(n_samples)),
        ("data5", create_data5(n_samples)),
        ("data6", create_data6(n_samples)),
        ("data7", create_data7(n_samples)),
    ]

    return datasets


def create_set_uci():
    datasets = [
        ("ecoli", create_ecoli()),
        ("glass", create_glass()),
        ("yeast", create_yeast()),
        ("statlog", create_statlog()),
        ("wdbc", create_wdbc()),
        ("wine", create_wine()),
    ]

    return datasets

def create_set_s():
    return [(f"s{i}", create_s(i)) for i in [1,2,3,4]]

def create_set_a():
    return [(f"a{i}", create_a(i)) for i in [1, 2, 3]]

def create_set_g(dims):
    return [(f"g{dims}_{i}", create_g(dims, i)) for i in [10,20,30,40,50,60,70,80,90]]



def create_aggregation():
    return read_data_and_labels(f"./data/sipu/aggregation.data", f"./data/sipu/aggregation.labels0")

def create_compound():
    return read_data_and_labels(f"./data/sipu/compound.data", f"./data/sipu/compound.labels0")

def create_d31():
    return read_data_and_labels(f"./data/sipu/d31.data", f"./data/sipu/d31.labels0")

def create_jain():
    return read_data_and_labels(f"./data/sipu/jain.data", f"./data/sipu/jain.labels0")

def create_pathbased():
    return read_data_and_labels(f"./data/sipu/pathbased.data", f"./data/sipu/pathbased.labels0")

def create_spiral():
    return read_data_and_labels(f"./data/sipu/spiral.data", f"./data/sipu/spiral.labels0")

def create_unbalance():
    return read_data_and_labels(f"./data/sipu/unbalance.data", f"./data/sipu/unbalance.labels0")


def create_set_sipu():
    datasets = [
        ("aggregation", create_aggregation()),
        ("compound", create_compound()),
        ("d31", create_d31()),
        ("jain", create_jain()),
        ("pathbased", create_pathbased()),
        ("spiral", create_spiral()),
        ("unbalance", create_unbalance()),
    ]

    return datasets

# n_samples = 1000
# dims = 2
# MAP_SETS_OF_DATA = {
#     f"simple{n_samples}": [
#         create_data1(n_samples),
#         create_data2(n_samples),
#         create_data3(n_samples),
#         create_data4(n_samples),
#         create_data5(n_samples),
#         create_data6(n_samples),
#         create_data7(n_samples),
#     ],
#
#     "s": [create_s(i) for i in [1,2,3,4]],
#     "a": [create_a(i) for i in [1, 2, 3]],
#     f"g{dims}": [create_g(dims, i) for i in [10,20,30,40,50,60,70,80,90]],
#
#     "uci": [
#         create_ecoli(),
#         create_glass(),
#         create_yeast(),
#         create_statlog(),
#         create_wdbc(),
#         create_wine(),
#     ]
#
# }

if __name__ == '__main__':
    X, y = create_data1(1000)
    print(X.shape, y.shape, len(np.unique(y)))

    # X, y = create_a(1)
    # print(X.shape, y.shape, len(np.unique(y)))
    #
    # X, y = create_a(2)
    # print(X.shape, y.shape, len(np.unique(y)))
    #
    # X, y = create_a(3)
    # print(X.shape, y.shape, len(np.unique(y)))

    X, y = create_g(2,10)
    print(X.shape, y.shape, len(np.unique(y)))

