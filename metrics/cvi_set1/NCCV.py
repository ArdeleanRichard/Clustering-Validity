import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

class NCCV_Index:
    def __init__(self, data, labels):
        """
        Parameters:
        """
        self.data = data
        self.labels = labels
        self.n = self.data.shape[0]
    
    def _compute_pairwise_vector(self, data):
        """
        Computes the pairwise Euclidean distances between rows in 'data'
        and returns the upper-triangular portion as a vector.
        """
        return pdist(data, metric='euclidean')
    
    def _compute_NC_for_partition(self, labels):
        """
        Given a clustering partition (labels for each data point), compute the 
        NC correlation defined as the Pearson correlation between the vector
        of pairwise distances (d) and the vector of distances between the centroids
        of the clusters to which the two points belong (c).
        """
        # Compute centroids for each cluster
        unique_labels = np.unique(labels)
        centroids = {lab: np.mean(self.data[labels == lab], axis=0) for lab in unique_labels}
        # For each data point, assign its centroid
        centroids_array = np.array([centroids[lab] for lab in labels])
        
        # Compute pairwise distance vectors (upper triangular part)
        d_vec = self._compute_pairwise_vector(self.data)
        c_vec = self._compute_pairwise_vector(centroids_array)
        
        # If either vector is constant, return 0 to avoid division by zero.
        if np.std(d_vec) == 0 or np.std(c_vec) == 0:
            return 0.0
        
        nc, _ = pearsonr(d_vec, c_vec)
        return nc
    
    def run(self):
        """
        Computes and returns the cluster validity index (NC index) for the given partition.
        For k = 1 (all points in one cluster), the index is defined as:
          NC = std(distance to global centroid) / (max - min distance)
        For k >= n (each point is its own cluster), NC is defined as 1.
        Otherwise, NC is the Pearson correlation between the vector of pairwise distances
        of the original data and that of the corresponding cluster centroids.
        
        Returns:
          A single float value representing the cluster validity index.
        """
        unique_clusters = np.unique(self.labels)
        k = len(unique_clusters)
        
        if k == 1:
            global_centroid = np.mean(self.data, axis=0)
            dists = np.linalg.norm(self.data - global_centroid, axis=1)
            r = np.max(dists) - np.min(dists)
            nc = (np.std(dists) / r) if r != 0 else 0.0
        elif k >= self.n:
            nc = 1.0
        else:
            nc = self._compute_NC_for_partition(self.labels)
        return nc