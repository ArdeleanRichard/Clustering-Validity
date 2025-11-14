import numpy as np
import math
from scipy.spatial.distance import cdist, euclidean


class CDbwIndex:
    def __init__(self, data, labels, metric="euclidean", r=20, shrink_factors=None):
        """
        Initialize the RevisedCDbwIndex class.
        
        Parameters:
        -----------
        metric : str, optional
            Distance metric to use (default is "euclidean").
        r : int, optional
            Number of representative points per cluster (recommended r>=10).
        shrink_factors : list or None, optional
            List of shrink factors in (0,1) for intra–cluster density calculation.
            Default is np.linspace(0.1, 0.8, 8).
        """
        self.data = data
        self.labels = labels
        self.metric = metric
        self.r = r
        if shrink_factors is None:
            self.shrink_factors = np.linspace(0.1, 0.8, 8)
        else:
            self.shrink_factors = shrink_factors

        # Validate cluster count (at least 2 clusters)
        unique_labels = np.unique(self.labels)
        if len(unique_labels) < 2 or len(unique_labels) > len(self.data)-1:
            raise ValueError("Number of unique labels must be >1 and < n_samples")

        # Precompute distance function (only euclidean is supported in this example)
        if self.metric != "euclidean":
            raise NotImplementedError("Currently only the 'euclidean' metric is supported.")
        
        # Compute per-cluster stats
        self.clusters = {label: self.data[self.labels == label] for label in unique_labels}
        self.cluster_centers = {label: np.mean(self.clusters[label], axis=0) for label in unique_labels}
        # Standard deviation per cluster (average std over dimensions)
        self.cluster_std = {
            label: math.sqrt(np.sum(np.std(self.clusters[label], axis=0)**2) / self.data.shape[1])
            for label in unique_labels
        }
        # Generate representatives for each cluster
        self.representatives = {label: self._compute_representatives(self.clusters[label], self.r)
                                for label in unique_labels}
    
    def _compute_representatives(self, X_cluster, r):
        """
        Compute r representative points for a cluster using a furthest-first technique.
        The first representative is chosen as the point farthest from the cluster center.
        Then, iteratively, choose the point with the maximum distance from all already selected reps.
        """
        if len(X_cluster) == 0:
            return np.array([])
        center = np.mean(X_cluster, axis=0)
        # Compute distances to center and choose the farthest as first rep.
        dists = np.linalg.norm(X_cluster - center, axis=1)
        first_rep = X_cluster[np.argmax(dists)]
        reps = [first_rep]
        for _ in range(1, min(r, len(X_cluster))):
            # For each point, compute distance to nearest already chosen rep
            d_to_reps = np.array([min(np.linalg.norm(x - rep) for rep in reps) for x in X_cluster])
            next_rep = X_cluster[np.argmax(d_to_reps)]
            reps.append(next_rep)
        return np.array(reps)
    
    def _shrink_representative(self, rep, center, s):
        """
        Shrink representative rep toward center by factor s.
        This returns: rep_shrunk = rep + s * (center - rep)
        """
        return rep + s * (center - rep)
    
    def intra_cluster_density(self):
        """
        Compute the intra-cluster density for each cluster over different shrink factors.
        For each cluster, and for each shrink factor, shrink each representative toward the cluster center
        and count the fraction of points in the cluster whose distance to the shrunk rep is less than the cluster's std.
        Returns:
            densities: dict mapping cluster label to list of density values (one per shrink factor)
        """
        densities = {}
        for label, X_cluster in self.clusters.items():
            center = self.cluster_centers[label]
            std_val = self.cluster_std[label]
            reps = self.representatives[label]
            density_s = []
            # For each shrink factor s, compute average density over representatives
            for s in self.shrink_factors:
                rep_densities = []
                for rep in reps:
                    rep_shrunk = self._shrink_representative(rep, center, s)
                    # Count number of points in cluster whose distance to rep_shrunk is less than std_val
                    dists = np.linalg.norm(X_cluster - rep_shrunk, axis=1)
                    count = np.sum(dists < std_val)
                    # Normalize by the total number of points in the cluster
                    rep_densities.append(count / len(X_cluster))
                density_s.append(np.mean(rep_densities))
            densities[label] = density_s
        return densities

    def compute_compactness(self, densities):
        """
        Compute the compactness (average intra-cluster density) over all clusters.
        According to Eq. 7, we average the density over the considered shrink factors and over clusters.
        """
        compactness_per_cluster = {label: np.mean(density_list) for label, density_list in densities.items()}
        overall_compactness = np.mean(list(compactness_per_cluster.values()))
        return overall_compactness, compactness_per_cluster

    def compute_intra_density_change(self, densities):
        """
        Compute the intra-density change for each cluster.
        According to Definition 8, we compute the sum of absolute differences of density between successive s values.
        Then average over clusters.
        """
        change_per_cluster = {}
        for label, density_list in densities.items():
            density_list = np.array(density_list)
            # Sum of absolute differences between successive shrink factors
            change = np.sum(np.abs(np.diff(density_list)))
            change_per_cluster[label] = change / (len(density_list) - 1)  # average change per step
        overall_change = np.mean(list(change_per_cluster.values()))
        return overall_change, change_per_cluster

    def compute_cohesion(self, compactness, intra_change):
        """
        Define cohesion as the sum of compactness and intra-cluster density change (Eq. 9).
        (Other formulations are possible; this is one interpretation.)
        """
        return compactness + intra_change

    def _mutual_closest_representatives(self, reps_i, reps_j):
        """
        Compute the mutual closest representatives between two clusters.
        For each representative in reps_i, find the closest in reps_j; then from reps_j, 
        find the closest in reps_i. The mutual set is the intersection.
        Returns a list of pairs (rep_i, rep_j) that are mutual.
        """
        pairs = []
        # For each rep in i, find index of closest rep in j
        closest_in_j = np.argmin(cdist(reps_i, reps_j, metric=self.metric), axis=1)
        # For each rep in j, find index of closest rep in i
        closest_in_i = np.argmin(cdist(reps_j, reps_i, metric=self.metric), axis=1)
        for i, j in enumerate(closest_in_j):
            # Check mutual: if reps_i[i] is the closest to reps_j[j]
            if closest_in_i[j] == i:
                pairs.append((reps_i[i], reps_j[j]))
        return pairs

    def inter_cluster_density(self):
        """
        Compute the inter-cluster density measure (Sep) over all pairs of clusters.
        For each pair of clusters, using their mutual closest representatives,
        we compute a density measure based on the fraction of points (from both clusters)
        that lie in the neighborhood of the midpoint of the rep pair (distance < average of stds)
        normalized by the distance between the representatives.
        Finally, for each cluster, we take the maximum inter-density with respect to any other cluster,
        and then average over clusters.
        (This is one interpretation of Definitions 3-5.)
        """
        labels = np.unique(self.labels)
        inter_densities = []
        for i, li in enumerate(labels):
            max_density = 0
            for lj in labels:
                if li == lj:
                    continue
                reps_i = self.representatives[li]
                reps_j = self.representatives[lj]
                # Get mutual closest representatives
                pairs = self._mutual_closest_representatives(reps_i, reps_j)
                if not pairs:
                    continue
                # For each pair, compute the density measure
                pair_densities = []
                std_avg = (self.cluster_std[li] + self.cluster_std[lj]) / 2.0
                for rep_i, rep_j in pairs:
                    midpoint = (rep_i + rep_j) / 2.0
                    # Consider points from both clusters that are within std_avg of the midpoint.
                    X_i = self.clusters[li]
                    X_j = self.clusters[lj]
                    count_i = np.sum(np.linalg.norm(X_i - midpoint, axis=1) < std_avg)
                    count_j = np.sum(np.linalg.norm(X_j - midpoint, axis=1) < std_avg)
                    # Density for the pair: total count normalized by (distance between reps * total points)
                    dist = np.linalg.norm(rep_i - rep_j)
                    # Avoid division by zero.
                    if dist == 0:
                        pair_density = 0
                    else:
                        pair_density = (count_i + count_j) / (dist * (len(X_i) + len(X_j)))
                    pair_densities.append(pair_density)
                # Use maximum density among representative pairs for this cluster pair.
                inter_density = max(pair_densities) if pair_densities else 0
                if inter_density > max_density:
                    max_density = inter_density
            inter_densities.append(max_density)
        # Sep is the average maximum inter-cluster density over clusters.
        Sep = np.mean(inter_densities)
        return Sep

    def compute_index(self):
        """
        Compute the overall CDbw index.
        
        Step 1: Intra-cluster density is computed at multiple shrink factors.
        Step 2: Compute Compactness (Eq. 7) and intra-cluster density change (Eq. 8).
        Step 3: Cohesion is defined as Compactness + density change (Eq. 9).
        Step 4: Inter-cluster density (Sep) is computed from mutual closest representatives (Eq. 5).
        Step 5: Separation wrt compactness is defined as Sep * Compactness (Eq. 10).
        Finally, CDbw is the product of Separation and Cohesion (Eq. 11).
        
        Returns:
            CDbw_index : float
                The computed CDbw index value.
        """
        # Intra-cluster density computations:
        densities = self.intra_cluster_density()
        compactness, comp_dict = self.compute_compactness(densities)
        intra_change, change_dict = self.compute_intra_density_change(densities)
        cohesion = self.compute_cohesion(compactness, intra_change)
        
        # Inter-cluster separation measure
        Sep = self.inter_cluster_density()
        # Separation wrt compactness:
        SC = Sep * compactness
        
        # Final CDbw index:
        CDbw_index = SC * cohesion
        return CDbw_index

    def run(self):
        """
        Compute and return the full CDbw index.
        """
        return self.compute_index()