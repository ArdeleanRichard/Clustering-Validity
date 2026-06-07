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

from cvis_ours.ad_CVIs import ad_silhouette_score, ad_davies_bouldin_score, ad_calinski_harabasz_score, arboris_index

from cvis_ours.measures import imbalance_ratio, overlap_ratio


MAP_METRIC_TO_FUNCTION = {
    # # CVI metrics
    "cSIL": lambda data, labels: cvi.cSIL().get_cvi(data, labels),
    "GD43": lambda data, labels: cvi.GD43().get_cvi(data, labels),
    "GD53": lambda data, labels: cvi.GD53().get_cvi(data, labels),
    "PS": lambda data, labels: cvi.PS().get_cvi(data, labels),
    "rCIP": lambda data, labels: cvi.rCIP().get_cvi(data, labels),
    "WB": lambda data, labels: cvi.WB().get_cvi(data, labels),
    "XB": lambda data, labels: cvi.XB().get_cvi(data, labels),

    # Permetrics metrics
    "SSE": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).sum_squared_error_index(),
    "RS": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).r_squared_index(),
    "DH": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).duda_hart_index(),
    "B": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).beale_index(),
    "BH": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).ball_hall_index(),
    "D": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).dunn_index(),
    "H": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).hartigan_index(),

    "DBCV": lambda X, labels: dbcv(X=X, y=labels),

    "I": lambda X, labels: i_index(X=X, labels=labels),

    "C": lambda X, labels: c_index(X=X, labels=labels),

    "CDbw": lambda X, labels: CDbw(X=X, labels=labels),
    "VIASCKDE": lambda X, labels: VIASCKDE(X=X, labels=labels),

    "COP": lambda X, labels: cop(data=X, labels=labels),
    "Sym": lambda X, labels: gSym(data=X, labels=labels).Sym(),
    "CS": lambda X, labels: cs_index(X=X, labels=labels),


    # # PyCVI metrics
    "SF": lambda data, labels: pycvi_cvi.ScoreFunction()(data, get_clustering(labels)),
    "SD": lambda data, labels: pycvi_cvi.SD()(data, get_clustering(labels)),
    "SDbw": lambda data, labels: pycvi_cvi.SDbw()(data, get_clustering(labels)),
    "XB*": lambda data, labels: pycvi_cvi.XBStar()(data, get_clustering(labels)),

    # # sklearn metrics
    "S": silhouette_score,
    "DB": davies_bouldin_score,
    "CH": calinski_harabasz_score,

    # #### our metrics
    # # # # "ED-S": ed_silhouette_score,
    # # # # "ED-DB": ed_davies_bouldin_score,
    # # # # "ED-CH": ed_calinski_harabasz_score,

    "AD-S": ad_silhouette_score,
    "AD-DB": ad_davies_bouldin_score,
    "AD-CH": ad_calinski_harabasz_score,

    "AD-idea": arboris_index,
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
    "ad-db"
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
    "arboris_index": ("AD-Idea", arboris_index),
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

MAP_LABELSET_TO_NAME = {
    "gt": "Ground Truth labels (TL)",
    "rl": "Random labels (RL)",
    "dfl": "First Diagonal separated labels (FDL)",
    "dsl": "Second Diagonal separated labels (SDL)",
    "vl": "Vertical midline separated labels (VL)",
    "hl": "Horizontal midline separated labels (HL)",
}