import numpy as np

from constants import LABEL_COLOR_MAP, FOLDER_RESULTS, FOLDER_FIGS_DATA, scale
from load_labelsets import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line


def remove_dups(X, gt):
    uniq_rows, idx = np.unique(X, axis=0, return_index=True)
    keep_idx = np.sort(idx)  # sort to preserve original order of first occurrences
    X = X[keep_idx]
    gt = gt[keep_idx]
    return X, gt

def reencode(labels):
    unique_labels, encoded = np.unique(labels, return_inverse=True)
    return encoded


def load_labelsets(X, gt, scale, label_sets, list_labelsets):
    midpoint = np.mean(scale)

    # Generate label sets
    if "dfl" in list_labelsets:
        dfl = assign_labels_by_given_line(X, diagonal_line(X, "first"))
        label_sets["dfl"] = dfl
    if "dsl" in list_labelsets:
        dsl = assign_labels_by_given_line(X, diagonal_line(X, "second"))
        label_sets["dsl"] = dsl
    if "vl" in list_labelsets:
        vl = assign_labels_by_given_line(X, vertical_line(midpoint))
        label_sets["vl"] = vl
    if "hl" in list_labelsets:
        hl = assign_labels_by_given_line(X, horizontal_line(midpoint))
        label_sets["hl"] = hl
    if "rl" in list_labelsets:
        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))
        label_sets["rl"] = rl

    return label_sets


def choose_colors(labels):
    label_color = [LABEL_COLOR_MAP[i] for i in labels]

    return label_color
