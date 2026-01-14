import numpy as np
from sklearn import datasets
from sklearn.datasets import make_blobs

from constants import random_state


def cluster_stats(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)
    stats = {}

    for label in np.unique(labels):
        cluster_points = X[labels == label]
        n = len(cluster_points)
        mean = cluster_points.mean(axis=0)
        cov = np.cov(cluster_points, rowvar=False)

        stats[int(label)] = {
            "n": int(n),
            "mean": mean.tolist(),
            "cov": cov.tolist()
        }

    return stats

def obtain_UNBALANCE_STATS():
    X, labels = create_unbalance()
    stats = cluster_stats(X, labels)

    for k, v in stats.items():
        print(f"    {k}: {v},")


def load_UNBALANCE_STATS():
    UNBALANCE_STATS = {
        0: {'n': 2000, 'mean': [150006.7365, 350103.876], 'cov': [[9982716.372253874, 1193299.6986753377], [1193299.6986753377, 10042633.918583289]]},
        1: {'n': 2000, 'mean': [179954.98, 380007.9705], 'cov': [[9869979.581390698, 1179290.3195697847], [1179290.3195697847, 9674416.364812154]]},
        2: {'n': 2000, 'mean': [209948.245, 349963.26], 'cov': [[9665368.451200599, 517801.0848424211], [517801.0848424211, 9168240.486643318]]},
        3: {'n': 100, 'mean': [440754.33, 298283.2], 'cov': [[85839933.63747482, -2361146.793939391], [-2361146.793939391, 89169344.6868687]]},
        4: {'n': 100, 'mean': [440134.41, 400135.41], 'cov': [[120445372.18373743, 4221838.76959596], [4221838.76959596, 125848151.05242425]]},
        5: {'n': 100, 'mean': [491036.01, 349798.33], 'cov': [[107976910.91909094, -3136604.417474747], [-3136604.417474747, 97399620.34454548]]},
        6: {'n': 100, 'mean': [539379.19, 299652.83], 'cov': [[97408312.98373738, 1546637.083131312], [1546637.083131312, 94213029.71828282]]},
        7: {'n': 100, 'mean': [538883.52, 400947.36], 'cov': [[93971552.77737373, 6441541.154343435], [6441541.154343435, 75038436.87919189]]},
    }

    # Identify minority clusters (5 smallest)
    MINORITY_IDS = sorted(UNBALANCE_STATS.keys(), key=lambda k: UNBALANCE_STATS[k]['n'])[:5]
    MAJORITY_IDS = [k for k in UNBALANCE_STATS.keys() if k not in MINORITY_IDS]

    return UNBALANCE_STATS, MAJORITY_IDS, MINORITY_IDS


def generate_unbalance_like(UNBALANCE_STATS, MAJORITY_IDS, MINORITY_IDS, scale_minority=1.0):
    """Generate data similar to unbalance.csv with option to scale minority clusters."""

    X_parts, y_parts = [], []
    for cid, stats in UNBALANCE_STATS.items():
        n = stats['n']
        if cid in MINORITY_IDS:
            n = int(n * scale_minority)
        mean = np.array(stats['mean'])
        cov = np.array(stats['cov'])
        Xi = np.random.multivariate_normal(mean, cov, size=n)
        yi = np.full(n, cid)
        X_parts.append(Xi)
        y_parts.append(yi)
    print(f"Minority Scale {scale_minority}: {np.unique(np.concatenate(y_parts), return_counts=True)}")
    return np.vstack(X_parts), np.concatenate(y_parts)

def generate_clusters_analysis(centers, sizes, cluster_std=1.0):
    """
    Generate a dataset from Gaussian blobs centered at `centers` with sample counts `sizes`.
    Returns (X, labels) where X is (N, 2) and labels is length N.
    """
    centers = np.asarray(centers)
    # X_parts = []
    # labels_parts = []
    # for i, (c, n) in enumerate(zip(centers, sizes)):
    #     X_i = np.random.normal(loc=0.0, scale=cluster_std, size=(n, centers.shape[1])) + np.asarray(c)
    #     X_parts.append(X_i)
    #     labels_parts.append(np.full(n, i, dtype=int))
    # X = np.vstack(X_parts)
    # labels = np.concatenate(labels_parts)

    X, labels = make_blobs(n_samples=list(map(int, sizes)),
                           centers=centers,
                           cluster_std=cluster_std,
                           random_state=random_state,
                           shuffle=False)

    return X, labels





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
    return datasets.make_blobs(n_samples=n_samples, random_state=random_state)

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
    return datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)

def create_data7(n_samples):
    return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)



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


def create_unbalance():
    return read_data_and_labels(f"./data/sipu/unbalance.data", f"./data/sipu/unbalance.labels0")


def create_d31():
    return read_data_and_labels(f"./data/sipu/d31.data", f"./data/sipu/d31.labels0")

def create_set_s():
    return [(f"s{i}", create_s(i)) for i in [1,2,3,4]]

def create_set_a():
    return


def create_trajectories():
    return read_data_and_labels(f"./data/wut/trajectories.data", f"./data/wut/trajectories.labels0")

def create_hepta():
    return read_data_and_labels(f"./data/fcps/hepta.data", f"./data/fcps/hepta.labels0")

def create_tetra():
    return read_data_and_labels(f"./data/fcps/tetra.data", f"./data/fcps/tetra.labels0")






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


def create_ecoli():
    return read_data_and_labels(f"./data/uci/ecoli.data", f"./data/uci/ecoli.labels0")

def create_glass():
    return read_data_and_labels(f"./data/uci/glass.data", f"./data/uci/glass.labels0")

def create_sonar():
    return read_data_and_labels(f"./data/uci/sonar.data", f"./data/uci/sonar.labels0")

def create_ionosphere():
    return read_data_and_labels(f"./data/uci/ionosphere.data", f"./data/uci/ionosphere.labels0")

def create_statlog():
    return read_data_and_labels(f"./data/uci/statlog.data", f"./data/uci/statlog.labels0")

def create_wdbc():
    return read_data_and_labels(f"./data/uci/wdbc.data", f"./data/uci/wdbc.labels0")

def create_wine():
    return read_data_and_labels(f"./data/uci/wine.data", f"./data/uci/wine.labels0")

def create_yeast():
    return read_data_and_labels(f"./data/uci/yeast.data", f"./data/uci/yeast.labels0")

def create_ccbr():
    return read_data_and_labels(f"./data/uci/ccbr.data", f"./data/uci/ccbr.labels0")

def create_wholesale():
    return read_data_and_labels(f"./data/uci/wholesale.data", f"./data/uci/wholesale.labels0")

def create_seeds():
    return read_data_and_labels(f"./data/uci/seeds.data", f"./data/uci/seeds.labels0")





def create_s(n=1):
    # data, meta = arff.loadarff(f'./data/s/s-set{n}.arff')
    # return transform_arff_data(data)
    return read_data_and_labels(f"./data/s/s{n}.data", f"./data/s/s{n}.labels0")

def create_a(n=1):
    return read_data_and_labels(f"./data/a/a{n}.data", f"./data/a/a{n}.labels0")

def create_ring(type):
    return read_data_and_labels(f"./data/graves/ring{type}.data", f"./data/graves/ring{type}.labels0")

def create_zigzag(type):
    return read_data_and_labels(f"./data/graves/zigzag{type}.data", f"./data/graves/zigzag{type}.labels0")



def create_parabolic():
    return read_data_and_labels(f"./data/graves/parabolic.data", f"./data/graves/parabolic.labels0")


def create_line():
    return read_data_and_labels(f"./data/graves/line.data", f"./data/graves/line.labels0")

def create_dense():
    return read_data_and_labels(f"./data/graves/dense.data", f"./data/graves/dense.labels0")

def create_fuzzyx():
    return read_data_and_labels(f"./data/graves/fuzzyx.data", f"./data/graves/fuzzyx.labels0")

def create_x(n):
    return read_data_and_labels(f"./data/wut/x{n}.data", f"./data/wut/x{n}.labels0")

def create_mk(n):
    return read_data_and_labels(f"./data/wut/mk{n}.data", f"./data/wut/mk{n}.labels0")

def create_smile():
    return read_data_and_labels(f"./data/wut/smile.data", f"./data/wut/smile.labels0")







def create_aggregation():
    return read_data_and_labels(f"./data/sipu/aggregation.data", f"./data/sipu/aggregation.labels0")

def create_compound():
    return read_data_and_labels(f"./data/sipu/compound.data", f"./data/sipu/compound.labels0")


def create_jain():
    return read_data_and_labels(f"./data/sipu/jain.data", f"./data/sipu/jain.labels0")

def create_pathbased():
    return read_data_and_labels(f"./data/sipu/pathbased.data", f"./data/sipu/pathbased.labels0")

def create_spiral():
    return read_data_and_labels(f"./data/sipu/spiral.data", f"./data/sipu/spiral.labels0")

def create_r15():
    return read_data_and_labels(f"./data/sipu/r15.data", f"./data/sipu/r15.labels0")

def create_flame():
    return read_data_and_labels(f"./data/sipu/flame.data", f"./data/sipu/flame.labels0")





def create_target():
    return read_data_and_labels(f"./data/fcps/target.data", f"./data/fcps/target.labels0")

def create_twodiamonds():
    return read_data_and_labels(f"./data/fcps/twodiamonds.data", f"./data/fcps/twodiamonds.labels0")

def create_wingnut():
    return read_data_and_labels(f"./data/fcps/wingnut.data", f"./data/fcps/wingnut.labels0")

def create_lsun():
    return read_data_and_labels(f"./data/fcps/lsun.data", f"./data/fcps/lsun.labels0")






def create_real_datasets():
    datasets = []
    datasets.append(("ecoli", create_ecoli()))
    datasets.append(("glass", create_glass()))
    datasets.append(("ionosphere", create_ionosphere()))
    datasets.append(("sonar", create_sonar()))
    datasets.append(("statlog", create_statlog()))
    datasets.append(("wdbc", create_wdbc()))
    datasets.append(("wine", create_wine()))
    datasets.append(("yeast", create_yeast()))
    # datasets.append(("ccbr", create_ccbr()))
    # datasets.append(("wholesale", create_wholesale()))
    # datasets.append(("seeds", create_seeds()))
    return datasets




def create_synthetic_datasets():
    datasets = []

    # graves
    datasets.append(("fuzzyx", create_fuzzyx()))
    datasets.append(("line", create_line()))
    datasets.append(("dense", create_dense()))
    datasets.append(("parabolic", create_parabolic()))
    datasets.extend([(f"ring{t}", create_ring(t)) for t in ["", "_noisy", "_outliers"]])
    datasets.extend([(f"zigzag{t}", create_zigzag(t)) for t in ["", "_noisy", "_outliers"]])

    # # wut
    datasets.extend([(f"mk{i}", create_mk(i)) for i in [1, 2]])  ### [1,2,3,4] ### >2 n_dims
    datasets.append((f"smile", create_smile()))
    datasets.extend([(f"x{i}", create_x(i)) for i in [1, 2, 3]])
    # ### datasets.extend([("trajectories", create_trajectories())]) ### high n_samples

    # # sipu
    datasets.append(("aggregation", create_aggregation()))
    datasets.append(("compound", create_compound()))
    datasets.append(("jain", create_jain()))
    datasets.append(("pathbased", create_pathbased()))
    datasets.append(("spiral", create_spiral()))
    datasets.append(("r15", create_r15()))
    datasets.append(("flame", create_flame()))
    # ### datasets.append(("d31", create_d31())) ### high n_samples
    # ### datasets.append(("unbalance", create_unbalance())) ### high n_samples
    # ### datasets.extend([(f"s{i}", create_s(i)) for i in [1,2,3,4]]) ### high n_samples
    # ### datasets.extend([(f"a{i}", create_a(i)) for i in [1, 2, 3]]) ### high n_samples

    # # fcps
    datasets.append(("lsun", create_lsun()))
    datasets.append(("target", create_target()))
    datasets.append(("twodiamonds", create_twodiamonds()))
    datasets.append(("wingnut", create_wingnut()))
    # ### datasets.append(("hepta", create_hepta())) ### >2 n_dims
    # ### datasets.append(("tetra", create_tetra())) ### >2 n_dims

    return datasets



if __name__ == '__main__':
    pass

