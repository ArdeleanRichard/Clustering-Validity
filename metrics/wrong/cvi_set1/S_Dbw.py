from scipy.spatial.distance import euclidean
import numpy as np


class S_Dbw_Index:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def run(self):
        # Unique clusters and number of clusters.
        unique_labels = np.unique(self.labels)
        c = len(unique_labels)
        
        # Compute overall dispersion σ(S): mean of per-dimension stds
        sigma_S = np.mean(np.std(self.data, axis=0))
        
        # For each cluster, compute its centroid and standard deviation (dispersion).
        centroids = {}
        cluster_std = {}
        densities_center = {}
        clusters = {}
        for label in unique_labels:
            idx = np.where(self.labels == label)[0]
            cluster_points = self.data[idx]
            clusters[label] = cluster_points
            centroids[label] = np.mean(cluster_points, axis=0)
            # Compute dispersion as the mean of per-dimension std for the cluster.
            cluster_std[label] = np.mean(np.std(cluster_points, axis=0))
            # Density at the cluster center: number of points in cluster with distance <= cluster_std[label]
            densities_center[label] = np.sum([euclidean(x, centroids[label]) <= cluster_std[label]
                                            for x in cluster_points])
        
        # Scatter term: average ratio of cluster dispersion over overall dispersion.
        scat = np.mean([cluster_std[label] / sigma_S for label in unique_labels])
        
        # Define the global radius for density computations as the average cluster std.
        global_radius = np.mean(list(cluster_std.values()))
        
        # Compute Dens_bw: for each pair of clusters.
        density_ratios = []
        # For each unique pair (i,j)
        for i in range(c):
            for j in range(i+1, c):
                label_i = unique_labels[i]
                label_j = unique_labels[j]
                centroid_i = centroids[label_i]
                centroid_j = centroids[label_j]
                # Midpoint between the two centroids.
                midpoint = (centroid_i + centroid_j) / 2.0
                
                # Density at the midpoint: count points in cluster i and cluster j
                density_mid = 0
                for point in clusters[label_i]:
                    if euclidean(point, midpoint) <= global_radius:
                        density_mid += 1
                for point in clusters[label_j]:
                    if euclidean(point, midpoint) <= global_radius:
                        density_mid += 1
                
                # Normalization factor: maximum density at the cluster centers.
                norm = max(densities_center[label_i], densities_center[label_j])
                # Avoid division by zero.
                if norm == 0:
                    ratio = 0
                else:
                    ratio = density_mid / norm
                density_ratios.append(ratio)
        
        if len(density_ratios) > 0:
            dens_bw = np.mean(density_ratios)
        else:
            dens_bw = 0
        
        # Final S_Dbw index.
        S_Dbw = scat + dens_bw
        return S_Dbw