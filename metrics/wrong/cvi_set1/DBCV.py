import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp


class DBCV_Index:
    def __init__(self, data, labels, noise_label=-1, dist_function=euclidean, inv_dist_clip=1e10):
        """
        Parameters:
          noise_label : label value indicating noise (default: -1).
          dist_function : callable
             Distance metric (default: Euclidean).
          inv_dist_clip : float
             Maximum allowed inverse distance to avoid overflow.
        """
        self.data = data
        self.labels = labels
        self.dist_function = dist_function
        self.n, self.dim = self.data.shape
        self.noise_label = noise_label
        self.inv_dist_clip = inv_dist_clip

    def run(self) -> float:
        """
        Compute the DBCV validity index in [-1,1]. Higher is better.
        """
        # Separate noise from valid clusters
        noise_mask = (self.labels == self.noise_label)
        valid_mask = ~noise_mask
        clusters = np.unique(self.labels[valid_mask])
        if clusters.size == 0:
            return 0.0

        cluster_data = {}
        # Phase 1: compute intra-cluster metrics
        for c in clusters:
            idx = np.where(self.labels == c)[0]
            Xc = self.data[idx]
            nc = len(idx)
            # Single-point clusters have VC = 0
            if nc < 2:
                cluster_data[c] = {'n': nc, 'VC': 0.0}
                continue

            # Pairwise distances
            D = cdist(Xc, Xc, metric=self.dist_function)

            # Core distances (vectorized per-point)
            core = np.zeros(nc)
            for i in range(nc):
                di = np.delete(D[i], i)
                inv_di = 1.0 / (di + 1e-10)
                inv_di = np.clip(inv_di, None, self.inv_dist_clip)
                logv = self.dim * np.log(inv_di)
                core[i] = np.exp(- (1.0 / self.dim) * (logsumexp(logv) - np.log(nc - 1)))

            # Mutual reachability distances (vectorized)
            ci = core[:, None]
            cj = core[None, :]
            mrd = np.maximum(np.maximum(ci, cj), D)
            np.fill_diagonal(mrd, 0.0)

            # Minimum spanning tree
            mst = minimum_spanning_tree(mrd).toarray()
            mst = np.maximum(mst, mst.T)

            # Identify internal nodes (degree > 1)
            degree = np.sum(mst > 0, axis=0)
            internal = np.where(degree > 1)[0]
            if internal.size == 0:
                internal = np.arange(nc)

            # Density Sparseness: max edge among internal nodes
            I = np.ix_(internal, internal)
            vals = mst[I]
            internal_vals = vals[vals > 0]
            DSC = np.max(internal_vals) if internal_vals.size else np.max(mst)

            cluster_data[c] = {
                'n': nc,
                'core': core,
                'internal': internal,
                'X': Xc,
                'DSC': DSC
            }

        # Phase 2: density separation & validity
        overall_score = 0.0
        for c in clusters:
            data_i = cluster_data[c]
            if data_i['n'] < 2:
                continue

            DSC = data_i['DSC']
            internal_i = data_i['internal']
            Xi_int = data_i['X'][internal_i]
            ci = data_i['core'][internal_i]

            dspc_vals = []
            for oc in clusters:
                if oc == c or cluster_data[oc]['n'] < 2:
                    continue
                data_j = cluster_data[oc]
                internal_j = data_j['internal']
                Xj_int = data_j['X'][internal_j]
                cj = data_j['core'][internal_j]

                D_between = cdist(Xi_int, Xj_int, metric=self.dist_function)
                civ = ci[:, None]
                cjv = cj[None, :]
                mrd_between = np.maximum(np.maximum(civ, cjv), D_between)
                dspc_vals.append(np.min(mrd_between))

            min_dspc = np.min(dspc_vals) if dspc_vals else 0.0
            denom = max(min_dspc, DSC)
            vc = (min_dspc - DSC) / denom if denom > 0 else 0.0

            weight = data_i['n'] / self.n
            overall_score += weight * vc

        # No extra noise penalty: weighted sum over all points already accounts for noise implicitly
        return overall_score
