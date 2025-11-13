import os

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

os.makedirs(FOLDER_RESULTS, exist_ok=True)
os.makedirs(FOLDER_FIGS_DATA, exist_ok=True)


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import cvi
from permetrics import ClusteringMetric
from pycvi import cvi as pycvi_cvi
from pycvi.cluster import get_clustering
from ed_scores import ed_silhouette_score, ed_davies_bouldin_score, ed_calinski_harabasz_score
from mst_scores import mst_silhouette_score, mst_davies_bouldin_score, mst_calinski_harabasz_score, mst_idea

from metrics.DBCV import DBCV
from metrics.VIASCKDE import VIASCKDE
from metrics.c_index import c_index
from metrics.i_index import i_index
from metrics.cvi_set1.DBCV import DBCV_Index
from metrics.cvi_set1.CDbw import CDbwIndex
from metrics.cvi_set1.NCCV import NCCV_Index
from metrics.cvi_set1.XieBeni import XieBeniIndex
from metrics.cvi_set1.S_Dbw import S_Dbw_Index
from metrics.cvi_set2.c_index import c_index2
from metrics.cvi_set2.cop import cop
from metrics.cvi_set2.dunn_index import dunn_index
from metrics.cvi_set2.gDunn import GeneralizedDunn
from metrics.cvi_set2.gSym import gSym
from metrics.cvi_set2.sdbw import SDbw

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
    "SSEI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).sum_squared_error_index(),
    "RSI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).r_squared_index(),
    "DHI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).duda_hart_index(),
    "BI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).beale_index(),
    "BHI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).ball_hall_index(),
    "DI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).dunn_index(),
    "HI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).hartigan_index(),
    "DBCVI": lambda X, labels: ClusteringMetric(X=X, y_pred=labels).density_based_clustering_validation_index(),

    # FOUND
    "DBCV": lambda X, labels: DBCV(X=X, labels=labels),
    "VIASCKDE": lambda X, labels: VIASCKDE(X=X, labels=labels),
    "C": lambda X, labels: c_index(X=X, labels=labels),
    "I": lambda X, labels: i_index(X=X, labels=labels),

    "DBCV2": lambda X, labels: DBCV_Index(data=X, labels=labels).run(),
    "CDbw": lambda X, labels: CDbwIndex(data=X, labels=labels).run(),
    "NCCV": lambda X, labels: NCCV_Index(data=X, labels=labels).run(),
    "SDbw2": lambda X, labels: S_Dbw_Index(data=X, labels=labels).run(),
    "XB2": lambda X, labels: XieBeniIndex(data=X, labels=labels).run(),

    "C2": lambda X, labels: c_index2(data=X, labels=labels),
    "COP": lambda X, labels: cop(data=X, labels=labels),
    "Dunn2": lambda X, labels: dunn_index(data=X, labels=labels),
    "GD43-2": lambda X, labels: GeneralizedDunn(data=X, labels=labels).generalized_exp(bd=4, sd=3),
    "GD53-2": lambda X, labels: GeneralizedDunn(data=X, labels=labels).generalized_exp(bd=5, sd=3),
    "Sym": lambda X, labels: gSym(data=X, labels=labels).Sym(),
    "SDbw3": lambda X, labels: SDbw(data=X, labels=labels).score(),




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
    "SS": silhouette_score,
    "DBS": davies_bouldin_score,
    "CHS": calinski_harabasz_score,

    # our metrics
    "ED-SS": ed_silhouette_score,
    "ED-DBS": ed_davies_bouldin_score,
    "ED-CHS": ed_calinski_harabasz_score,
    "MST-SS": mst_silhouette_score,
    "MST-DBS": mst_davies_bouldin_score,
    "MST-CHS": mst_calinski_harabasz_score,
    "idea": mst_idea,
}

METRICS = list(MAP_METRIC_TO_FUNCTION.keys())


# Define which metrics are "lower is better"
MAP_LOWER_IS_BETTER = {
    # CVI
    "rcip", "wb", "xb",

    # Permetrics
    "ssei", "bhi", "dbcvi",

    # PyCVI
    "sd", "sdbw", "xb*",

    # sklearn
    "dbs",

    # ours
    "ed-dbs",
    "mst-dbs",
    "idea",
}