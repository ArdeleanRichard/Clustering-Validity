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


target_path = "C:/WORK/Clustering-Validity/"
current_path = os.getcwd()

if os.path.abspath(current_path) == os.path.abspath(target_path):
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




