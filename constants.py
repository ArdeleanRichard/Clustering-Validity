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

LABEL_COLOR_MAP_LARGE = {
    -1: 'gray',
    0: 'red',
    1: 'mediumblue',
    2: 'green',
    3: 'yellow',
    4: 'magenta',
    5: 'cyan',
    6: 'wheat',
    7: 'yellowgreen',
    8: 'orchid',
    9: 'tab:orange',
    10: 'tab:brown',
    11: 'tab:pink',
    12: 'lime',
    13: 'silver',
    14: 'khaki',
    15: 'lightgreen',
    16: 'orangered',
    17: 'salmon',
    18: 'tab:purple',
    19: 'turquoise',
    20: 'royalblue',
    21: 'beige',
    22: 'crimson',
    23: 'indigo',
    24: 'darkblue',
    25: 'gold',
    26: 'ivory',
    27: 'lavender',
    28: 'lightblue',
    29: 'olive',
    30: 'sienna',
    31: 'darkgreen',
    32: 'darkred',
    33: 'darkorange',
    34: 'pink',
    35: 'purple',
    36: 'lightpink',
    37: 'lightyellow',
    38: 'lightcyan',
    39: 'lightgray',
    40: 'navy',
    41: 'teal',
    42: 'aqua',
    43: 'chartreuse',
    44: 'coral',
    45: 'orchid',
    46: 'peru',
    47: 'plum',
    48: 'salmon',
    49: 'sandybrown',
    50: 'seagreen',
    51: 'slateblue',
    52: 'tan',
    53: 'thistle',
    54: 'tomato',
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




