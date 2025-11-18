import os
import numpy as np

random_state = 42
np.random.seed(random_state)


LABEL_COLOR_MAP = {
    -1: 'gray',
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'yellow',
    4: 'magenta',
    5: 'cyan',
    6: 'wheat',
    7: 'yellowgreen',
    8: 'orchid',
    9: 'tab:orange',
    10: 'tab:brown',
}


FOLDER_RESULTS = f"./results/"
FOLDER_FIGS_DATA = f"./figs/data/"
FOLDER_FIGS_ANALYSIS = f"./figs/analysis/"
FOLDER_FIGS_ANALYSIS_ESTIMATE = f"./figs/analysis/estimate_k/"
FOLDER_FIGS_ANALYSIS_EXTERNAL = f"./figs/analysis/external/"
FOLDER_FIGS_ANALYSIS_INTERNAL = f"./figs/analysis/internal/"

os.makedirs(FOLDER_RESULTS, exist_ok=True)
os.makedirs(FOLDER_FIGS_DATA, exist_ok=True)
os.makedirs(FOLDER_FIGS_ANALYSIS_ESTIMATE, exist_ok=True)
os.makedirs(FOLDER_FIGS_ANALYSIS_EXTERNAL, exist_ok=True)
os.makedirs(FOLDER_FIGS_ANALYSIS_INTERNAL, exist_ok=True)


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
import cvi
from permetrics import ClusteringMetric
from pycvi import cvi as pycvi_cvi
from pycvi.cluster import get_clustering

from metrics.DBCV import dbcv
from metrics.VIASCKDE import VIASCKDE
from metrics.c_index import c_index
from metrics.i_index import i_index
from metrics.cvi_set2.cop import cop
from metrics.cvi_set2.gSym import gSym
from metrics.cdbw import CDbw
from metrics.cs_index import cs_index

# from metrics.wrong.cvi_set1.DBCV import DBCV_Index
# from metrics.cvi_set2.c_index import c_index2

from ours.ed_scores import ed_silhouette_score, ed_davies_bouldin_score, ed_calinski_harabasz_score
from ours.mst_scores import mst_silhouette_score, mst_davies_bouldin_score, mst_calinski_harabasz_score, mst_idea


MAP_METRIC_TO_FUNCTION = {
    # CVI metrics
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
    # "DBCVI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).density_based_clustering_validation_index(), # WRONG

    # FOUND
    # "CDbw2": lambda X, labels: CDbwIndex(data=X, labels=labels).run(), #WRONG
    # "DBCV2": lambda X, labels: DBCV_Index(data=X, labels=labels).run(),  # WRONG
    # "SDbw2": lambda X, labels: S_Dbw_Index(data=X, labels=labels).run(), # WRONG
    # "XB2": lambda X, labels: XieBeniIndex(data=X, labels=labels).run(), # WRONG
    # "NCCV": lambda X, labels: NCCV_Index(data=X, labels=labels).run(),  # QUESTIONABLE

    # "DBCV": lambda X, labels: DBCV(X=X, labels=labels),  # WRONG
    "DBCV": lambda X, labels: dbcv(X=X, y=labels),

    "I": lambda X, labels: i_index(X=X, labels=labels),

    # "C2": lambda X, labels: c_index2(data=X, labels=labels), # SAME RESULT
    "C": lambda X, labels: c_index(X=X, labels=labels),

    "CDbw": lambda X, labels: CDbw(X=X, labels=labels),
    "VIASCKDE": lambda X, labels: VIASCKDE(X=X, labels=labels),


    # "GD43-2": lambda X, labels: GeneralizedDunn(data=X, labels=labels).generalized_exp(bd=4, sd=3), # NaNs
    # "GD53-2": lambda X, labels: GeneralizedDunn(data=X, labels=labels).generalized_exp(bd=5, sd=3), # NaNs
    # "SDbw3": lambda X, labels: SDbw(data=X, labels=labels).score(), # WRONG
    # "Dunn2": lambda X, labels: dunn_index(data=X, labels=labels), # WRONG
    "COP": lambda X, labels: cop(data=X, labels=labels),
    "Sym": lambda X, labels: gSym(data=X, labels=labels).Sym(),
    "CS": lambda X, labels: cs_index(X=X, labels=labels),




    # PyCVI metrics
    # "Calinski-Harabasz":  lambda data, labels: pycvi_cvi.CalinskiHarabasz()(data, get_clustering(labels)),
    # "Davies-Bouldin":     lambda data, labels: pycvi_cvi.DaviesBouldin()(data, get_clustering(labels)),
    # "Silhouette":         lambda data, labels: pycvi_cvi.Silhouette()(data, get_clustering(labels)),
    # "Hartigan":           lambda data, labels: pycvi_cvi.Hartigan()(data, get_clustering(labels)),
    # "Dunn":               lambda data, labels: pycvi_cvi.Dunn()(data, get_clustering(labels)),
    # "Xie-Beni":           lambda data, labels: pycvi_cvi.XieBeni()(data, get_clustering(labels)),
    # ERROR: #"GP":         lambda data, labels: pycvi_cvi.GapStatistic()(data, get_clustering(labels)),
    # ERROR: #"MB":         lambda data, labels: pycvi_cvi.MaulikBandyopadhyay()(data, get_clustering(labels)),
    "SF": lambda data, labels: pycvi_cvi.ScoreFunction()(data, get_clustering(labels)),
    "SD": lambda data, labels: pycvi_cvi.SD()(data, get_clustering(labels)),
    "SDbw": lambda data, labels: pycvi_cvi.SDbw()(data, get_clustering(labels)),
    "XB*": lambda data, labels: pycvi_cvi.XBStar()(data, get_clustering(labels)),

    # sklearn metrics
    "S": silhouette_score,
    "DB": davies_bouldin_score,
    "CH": calinski_harabasz_score,

    # our metrics
    "ED-S": ed_silhouette_score,
    "ED-DB": ed_davies_bouldin_score,
    "ED-CH": ed_calinski_harabasz_score,
    "MST-S": mst_silhouette_score,
    "MST-DB": mst_davies_bouldin_score,
    "MST-CH": mst_calinski_harabasz_score,
    # # "idea": mst_idea,
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
    "mst-db",
    "idea",
}



MAP_EXTERNAL_METRICS = {
    "ari": ("ARI", "Adjusted Rand Index", adjusted_rand_score),
    "ami": ("AMI", "Adjusted Mutual Information", adjusted_mutual_info_score),
}


MAP_INTERNAL_METRICS = {
    "silhouette": ("Silhouette Score", silhouette_score),
}
