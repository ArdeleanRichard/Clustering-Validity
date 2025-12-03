import numpy as np
import scipy.spatial.distance as dist
from sklearn.neighbors import KDTree
import warnings


class gSym:
    """
		Implements a new class of cluster metrics based on point symmetry distance, not Euclidean
		Each individual metric mostly modifies an existing CVI, eg. Dunn Index, Davies Bouldin,
		resulting in indices such as Sym-DB, Sym-Dunn Index, etc,
	"""

    def __init__(self, data, labels):
        if len(labels) != len(data):
            warnings.warn('Failed! Dimensions of data and cluster labels are unequal')
            return np.NaN

        self.data = np.array(data)
        self.labels = np.array(labels)
        self.unique_labels = np.unique(labels)
        self.K = len(self.unique_labels)

        # Pre-compute cluster memberships and centroids
        self.clusters = {}
        self.centroids = {}
        for k in self.unique_labels:
            mask = self.labels == k
            self.clusters[k] = self.data[mask]
            self.centroids[k] = np.mean(self.clusters[k], axis=0)

    @staticmethod
    def sym_point(point, centroid):
        """
			Reflected / Symmetrical point with respect to centroid

			Parameters
			---------
			point: numpy array
			centroid: numpy array

			Returns
			-------
			symmetrical point: numpy array of reflected point
		"""
        return 2 * centroid - point

    @staticmethod
    def ps_distance(point, centroid, cluster, n_neighbors=2, **kwargs):
        """
			Computes point symmetrical distance of point with respect to the centroid

			Parameters
			---------
			point: numpy array
			centroid: numpy array
			cluster: numpy array of cluster point belongs to
			n_neighbors: int, number of neighbors in KD-Tree
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			symmetrical point: numpy array of reflected point
		"""

        # Get x* — symmetrical point w.r.t centroid
        x_sym = gSym.sym_point(point, centroid)

        # KD-Tree Nearest Neighbor search
        tree = KDTree(cluster, leaf_size=2)
        distance, ind = tree.query(np.array([x_sym]), k=n_neighbors)
        d_sym = np.sum(distance) / n_neighbors

        # Intra-cluster distance from centroid
        intra_cdist = dist.cdist([point], [centroid], **kwargs)[0, 0]

        return d_sym * intra_cdist

    @staticmethod
    def ps_distance_batch(points, centroid, cluster, n_neighbors=2, **kwargs):
        """
			Batch computation of point symmetrical distances

			Parameters
			---------
			points: numpy array of points (n_points, n_features)
			centroid: numpy array
			cluster: numpy array of cluster points belong to
			n_neighbors: int, number of neighbors in KD-Tree
			**kwargs: See scipy.spatial.distance **kwargs

			Returns
			-------
			ps_distances: numpy array of ps distances for all points
		"""
        # Get symmetrical points for all at once
        x_sym = 2 * centroid - points

        # Single KD-Tree for all queries
        tree = KDTree(cluster, leaf_size=2)
        distances, _ = tree.query(x_sym, k=n_neighbors)
        d_sym = np.mean(distances, axis=1)

        # Intra-cluster distances from centroid (vectorized)
        intra_cdist = dist.cdist(points, [centroid], **kwargs)[:, 0]

        return d_sym * intra_cdist

    def Sym(self, **kwargs):
        """
			Sym-Index: Symmetrical variation of I-Index
			Computed as ratio of maximum inter-centroid distances,
			divided by sum of point symmetry distances multiplied by number of clusters

			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, maximum value represents good partition

		"""

        ps_distance_total = 0

        # Compute all point symmetry distances using batch processing
        for k in self.unique_labels:
            cluster_k = self.clusters[k]
            centroid_k = self.centroids[k]

            # Batch process all points in cluster
            ps_distances = gSym.ps_distance_batch(cluster_k, centroid_k, cluster_k, **kwargs)
            ps_distance_total += np.sum(ps_distances)

        # Compute all inter-centroid distances at once
        centroid_array = np.array([self.centroids[k] for k in self.unique_labels])
        inter_cdist = dist.pdist(centroid_array, **kwargs)

        return np.max(inter_cdist) / (self.K * ps_distance_total)

    def SymDB(self, **kwargs):
        """
			Sym-DB: Symmetry-Based Davies-Bouldin Index
			Computed as DB Index, with Scatter modified to be average sum of all
			point-symmetry distances within clusters

			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, minimum value represents good partition

		"""

        # Pre-compute average ps_distances for all clusters
        avg_ps_distances = {}
        for k in self.unique_labels:
            cluster_k = self.clusters[k]
            centroid_k = self.centroids[k]

            ps_distances = gSym.ps_distance_batch(cluster_k, centroid_k, cluster_k, **kwargs)
            avg_ps_distances[k] = np.mean(ps_distances)

        # Compute pairwise centroid distances once
        centroid_array = np.array([self.centroids[k] for k in self.unique_labels])
        pairwise_cdist = dist.squareform(dist.pdist(centroid_array, **kwargs))

        R = []
        for i, k in enumerate(self.unique_labels):
            for j, l in enumerate(self.unique_labels):
                if i < j:  # Only compute upper triangle
                    inter_cdist = pairwise_cdist[i, j]
                    R.append((avg_ps_distances[k] + avg_ps_distances[l]) / inter_cdist)

        return np.sum(R) / self.K

    def Sym33(self, **kwargs):

        """
			Sym-Index: Symmetrical variation of gD33 (Varied Dunn's Index)
			Cohesion estimator modified as sum of point symmetry distances for cluster,
			divided by Cluster size and multiplied by 2


			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, maximum value represents good partition

		"""

        intra_cdist = []
        inter_cdist = []

        # Pre-compute intra-cluster distances
        for k in self.unique_labels:
            cluster_k = self.clusters[k]
            centroid_k = self.centroids[k]

            # Batch process PS distances
            ps_distances = gSym.ps_distance_batch(cluster_k, centroid_k, cluster_k, **kwargs)
            intra_cdist.append((2 * np.sum(ps_distances)) / len(cluster_k))

        # Compute inter-cluster distances
        for i, k in enumerate(self.unique_labels):
            cluster_k = self.clusters[k]
            for j in range(i + 1, len(self.unique_labels)):
                l = self.unique_labels[j]
                cluster_l = self.clusters[l]

                # Get minimum between-cluster distance
                inter_cdist.append(gSym.small_delta(cluster_k, cluster_l, **kwargs))

        return np.min(inter_cdist) / np.max(intra_cdist)

    @staticmethod
    def small_delta(cluster_k, cluster_l, **kwargs):
        """
			Small delta computation of separation estimator for gD33 index

			Parameters
			---------
			cluster_k: numpy array of first cluster
			cluster_l: numpy array of second cluster
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	between-cluster distance between cluster_k and cluster_l
		"""
        n_k = len(cluster_k)
        n_l = len(cluster_l)

        pw_dist = dist.cdist(cluster_k, cluster_l, **kwargs)

        return np.sum(pw_dist) / (n_k * n_l)


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_wine
    import sklearn.cluster as cluster
    import time

    iris = load_iris()
    clustering = cluster.KMeans().fit(iris.data)
    labels = clustering.labels_

    start = time.time()
    gSym_obj = gSym(iris.data, labels)
    print('SymDB: ', gSym_obj.SymDB())
    print('Sym33: ', gSym_obj.Sym33())
    print('Sym: ', gSym_obj.Sym())
    end = time.time()
    print('Time: ', end - start)