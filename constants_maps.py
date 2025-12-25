from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score
import cvi
from permetrics import ClusteringMetric
from pycvi import cvi as pycvi_cvi
from pycvi.cluster import get_clustering

from cvis.DBCV_opt import dbcv
from cvis.VIASCKDE_opt import VIASCKDE
from cvis.c_index import c_index
from cvis.cdbw import CDbw
from cvis.cs_index import cs_index
from cvis.cvi_set.cop import cop
from cvis.cvi_set.gSym import gSym
from cvis.i_index import i_index
from cvis_ours.ed_CVIs import ed_silhouette_score, ed_davies_bouldin_score, ed_calinski_harabasz_score

from cvis_ours.ad_CVIs import ad_silhouette_score, ad_davies_bouldin_score, ad_calinski_harabasz_score, ad_idea

from cvis_ours.measures import imbalance_ratio, overlap_ratio


MAP_METRIC_TO_FUNCTION = {
    # # CVI metrics
    # "cSIL": lambda data, labels: cvi.cSIL().get_cvi(data, labels),
    # "GD43": lambda data, labels: cvi.GD43().get_cvi(data, labels),
    # "GD53": lambda data, labels: cvi.GD53().get_cvi(data, labels),
    # "PS": lambda data, labels: cvi.PS().get_cvi(data, labels),
    # "rCIP": lambda data, labels: cvi.rCIP().get_cvi(data, labels),
    # "WB": lambda data, labels: cvi.WB().get_cvi(data, labels),
    # "XB": lambda data, labels: cvi.XB().get_cvi(data, labels),
    #
    # # Permetrics metrics
    # "SSE": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).sum_squared_error_index(),
    # "RS": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).r_squared_index(),
    # "DH": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).duda_hart_index(),
    # "B": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).beale_index(),
    # "BH": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).ball_hall_index(),
    # "D": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).dunn_index(),
    # "H": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).hartigan_index(),
    # # "DBCVI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).density_based_clustering_validation_index(), # WRONG
    #
    # "DBCV": lambda X, labels: dbcv(X=X, y=labels),
    #
    # "I": lambda X, labels: i_index(X=X, labels=labels),
    #
    # "C": lambda X, labels: c_index(X=X, labels=labels),
    #
    # "CDbw": lambda X, labels: CDbw(X=X, labels=labels),
    # "VIASCKDE": lambda X, labels: VIASCKDE(X=X, labels=labels),
    #
    #
    #
    #
    # "COP": lambda X, labels: cop(data=X, labels=labels),
    # "Sym": lambda X, labels: gSym(data=X, labels=labels).Sym(),
    # "CS": lambda X, labels: cs_index(X=X, labels=labels),
    #
    #
    # # PyCVI metrics
    # # "Calinski-Harabasz":  lambda data, labels: pycvi_cvi.CalinskiHarabasz()(data, get_clustering(labels)),
    # # "Davies-Bouldin":     lambda data, labels: pycvi_cvi.DaviesBouldin()(data, get_clustering(labels)),
    # # "Silhouette":         lambda data, labels: pycvi_cvi.Silhouette()(data, get_clustering(labels)),
    # # "Hartigan":           lambda data, labels: pycvi_cvi.Hartigan()(data, get_clustering(labels)),
    # # "Dunn":               lambda data, labels: pycvi_cvi.Dunn()(data, get_clustering(labels)),
    # # "Xie-Beni":           lambda data, labels: pycvi_cvi.XieBeni()(data, get_clustering(labels)),
    # # ERROR: #"GP":         lambda data, labels: pycvi_cvi.GapStatistic()(data, get_clustering(labels)),
    # # ERROR: #"MB":         lambda data, labels: pycvi_cvi.MaulikBandyopadhyay()(data, get_clustering(labels)),
    # "SF": lambda data, labels: pycvi_cvi.ScoreFunction()(data, get_clustering(labels)),
    # "SD": lambda data, labels: pycvi_cvi.SD()(data, get_clustering(labels)),
    # "SDbw": lambda data, labels: pycvi_cvi.SDbw()(data, get_clustering(labels)),
    # "XB*": lambda data, labels: pycvi_cvi.XBStar()(data, get_clustering(labels)),
    #
    # # sklearn metrics
    # "S": silhouette_score,
    # "DB": davies_bouldin_score,
    # "CH": calinski_harabasz_score,
    #
    # # our metrics
    # "ED-S": ed_silhouette_score,
    # "ED-DB": ed_davies_bouldin_score,
    # "ED-CH": ed_calinski_harabasz_score,
    #
    #
    # "MST-S": mst_silhouette_score,
    # "MST-DB": mst_davies_bouldin_score,
    # "MST-CH": mst_calinski_harabasz_score,


    "idea": ad_idea,

    # "MSTog-S": mst_silhouette_score_first,
    # "MSTog-DB": mst_davies_bouldin_score_first,
    # "MSTog-CH": mst_calinski_harabasz_score_first,
    #
    # "MST3-S": mst_silhouette_score3,
    # "MST3-DB": mst_davies_bouldin_score3,
    # "MST3-CH": mst_calinski_harabasz_score3,

    # "sbcvi":  lambda data, labels: AdaptiveGridMST(data, labels, min_points=5, max_depth=20).compute_cluster_validity_score()['final_score']
}


METRICS = list(MAP_METRIC_TO_FUNCTION.keys())


# Define which metrics are "lower is better"
MAP_LOWER_IS_BETTER = {
    # CVI
    "rcip", "wb", "xb",

    # Permetrics
    "sse", "bh", "dh", "b", "h",

    # PyCVI
    "sd", "sdbw", "xb*",

    # sklearn
    "db",

    # others
    "c",
    "cs",
    "cop",

    # ours
    "ed-db",
    "mst-db"
}


MAP_EXTERNAL_METRICS = {
    "ari": ("ARI", "Adjusted Rand Index", adjusted_rand_score),
    "ami": ("AMI", "Adjusted Mutual Information", adjusted_mutual_info_score),
}


MAP_INTERNAL_METRICS = {
    "silhouette": ("Silhouette Score", silhouette_score),
    "ad_silhouette": ("AD-S", ad_silhouette_score),
    "ad_db": ("AD-CH", ad_calinski_harabasz_score),
    "ad_ch": ("AD-DB", ad_davies_bouldin_score),
    "ad_idea": ("AD-Idea", ad_idea),
}


MAP_MEASURES = {
    "imbalance": ("IR", "Imbalance Ratio", imbalance_ratio),
    "overlap": ("OR", "Overlap Ratio", overlap_ratio),
}


MAP_MEASURE_TO_VARIABLE = {
    "imbalance": "n_minority",
    "overlap": "distance"
}



MAP_LABELSET = {
    "hl": (
        "Horizontal",
        lambda X: (X[:, 1] > ((X[:, 1].min() + X[:, 1].max()) / 2.0)).astype(int),
        lambda X: ((X[:, 0].min(), (X[:, 1].min() + X[:, 1].max()) / 2.0), (X[:, 0].max(), (X[:, 1].min() + X[:, 1].max()) / 2.0))
    ),
    "vl": (
        "Vertical",
        lambda X: (X[:, 0] > ((X[:, 0].min() + X[:, 0].max()) / 2.0)).astype(int),
        lambda X: (((X[:, 0].min() + X[:, 0].max()) / 2.0, X[:, 1].min()), ((X[:, 0].min() + X[:, 0].max()) / 2.0, X[:, 1].max())),
    ),
}


