import os
import pandas as pd

from constants import FOLDER_RESULTS_CVIS_CACHE



# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(dataset_name, clusterer_name, suffix=""):
    """
    Return the path for a cache CSV.
    suffix="" -> CVI values (full)
    suffix="_nn" -> CVI values (no-noise)
    suffix="_ARI" / "_ARI_nn" / "_BARI" / "_BARI_nn" -> external metric vectors
    """
    fname = f"{dataset_name}_{clusterer_name}{suffix}.csv"
    return os.path.join(FOLDER_RESULTS_CVIS_CACHE, fname)


def _save_cvi_cache(dataset_name, clusterer_name, cvi_matrix, param_keys, suffix=""):
    """
    Save CVI (full) matrix to CSV.

    Parameters
    ----------
    cvi_matrix : dict[metric -> list]   length == len(param_keys)
    param_keys : list[str]   column names (one per parameterisation)
    """
    df = pd.DataFrame(cvi_matrix, index=param_keys).T   # rows=CVIs, cols=params
    df.to_csv(_cache_path(dataset_name, clusterer_name, suffix))



def _save_external_cache(dataset_name, clusterer_name, suffix, values, param_keys):
    """
    Save a single external metric vector (ARI / BARI / …) to CSV.

    The resulting CSV has rows=CVIs (only one row here labelled with the suffix)
    and cols=param_keys, so that shape matches the CVI cache exactly.
    """
    os.makedirs(FOLDER_RESULTS_CVIS_CACHE, exist_ok=True)
    df = pd.DataFrame([values], index=[suffix.lstrip("_")], columns=param_keys)
    df.to_csv(_cache_path(dataset_name, clusterer_name, suffix))


def _load_cvi_cache(dataset_name, clusterer_name, suffix=""):
    """
    Load CVI (full) cache.
    Returns (cvi_dict, param_keys) or None if cache missing / unreadable.
    cvi_dict : dict[metric -> list[float]]   ordered by param_keys
    """
    path = _cache_path(dataset_name, clusterer_name, suffix)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        param_keys = list(df.columns)
        cvi_dict = {metric: list(df.loc[metric]) for metric in df.index}
        return cvi_dict, param_keys
    except Exception as e:
        print(f"  [cache] WARNING: failed to read {path}: {e}")
        return None


def _load_external_cache(dataset_name, clusterer_name, suffix):
    """
    Load one external metric vector. Returns (values_list, param_keys) or None.
    """
    path = _cache_path(dataset_name, clusterer_name, suffix)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        param_keys = list(df.columns)
        values = list(df.iloc[0])
        return values, param_keys
    except Exception as e:
        print(f"  [cache] WARNING: failed to read {path}: {e}")
        return None


def _cvi_cache_exists(dataset_name, clusterer_name):
    """Return True if the full CVI cache CSV exists for this (dataset, clusterer)."""
    return os.path.exists(_cache_path(dataset_name, clusterer_name, ""))


def _ari_cache_exists(dataset_name, clusterer_name):
    return os.path.exists(_cache_path(dataset_name, clusterer_name, "_ARI"))


def _main_caches_exist(dataset_name, clusterer_name):
    """Return True only when all required cache files are present."""
    suffixes = ["", "_ARI"]
    return all(os.path.exists(_cache_path(dataset_name, clusterer_name, s)) for s in suffixes)


def _all_caches_exist(dataset_name, clusterer_name):
    suffixes = ["", "_nn", "_ARI", "_ARI_nn", "_BARI", "_BARI_nn"]
    return all(os.path.exists(_cache_path(dataset_name, clusterer_name, s)) for s in suffixes)