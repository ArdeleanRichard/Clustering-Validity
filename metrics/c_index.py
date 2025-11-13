"""
Implementation of CIndex
https://github.com/johnvorsten/py_cindex/tree/master

Cindex =
Sw − Smin
Smax − Smin , Smin ?= Smax, Cindex ∈ (0, 1), (6)

Smin = is the sum of the Nw smallest distances between all the pairs of points
in the entire data set (there are Nt such pairs);

Smax = is the sum of the Nw largest distances between all the pairs of points
in the entire data set.


Citation:

A General Statistical Framework for Assessing Categorical Clustering in Free Recall.” Psychological Bulletin, 83(6), 1072–1080


"""
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


"""
##### Here is the Nb-Clust implementation of the CIndex #####
See https://cran.r-project.org/web/packages/NbClust/index.html
Note these two lines (R Code) :
    Dmin = min(v_min)
    Dmax = max(v_max)
    result <- (DU - r * Dmin)/(Dmax * r - Dmin * r)

Indice.cindex <- function (d, cl)
{
    d <- data.matrix(d)
    DU <- 0
    r <- 0
    v_max <- array(1, max(cl))
    v_min <- array(1, max(cl))
    for (i in 1:max(cl)) {
        n <- sum(cl == i)

        if (n > 1) {
            t <- d[cl == i, cl == i]
            DU = DU + sum(t)/2
            v_max[i] = max(t)

            if (sum(t == 0) == n)
                v_min[i] <- min(t[t != 0])

            else v_min[i] <- 0
            r <- r + n * (n - 1)/2
        }
    }

    Dmin = min(v_min)
    Dmax = max(v_max)
    if (Dmin == Dmax)
        result <- NA
    else result <- (DU - r * Dmin)/(Dmax * r - Dmin * r)
    result

}


##### Here is the ClusterSim implementation of the CIndex #####
See https://rdrr.io/cran/clusterSim/src/R/index.C.r
Note these two lines :
	Dmin=sum(sort(ddist)[1:r])
	Dmax=sum(sort(ddist,decreasing = T)[1:r])
They include the whole distance array, which includes all permutations of
distances between points (instead of combinations). This means the high
end and low end are double counted? I dont think that is the correct way to
calculate C Index


index.C<-function(d,cl)
{
  ddist<-d
	d<-data.matrix(d)
	DU<-0
	r<-0
	for (i in 1:max(cl))
	{
	  t<-d[cl==i,cl==i]
		n<-sum(cl==i)
		if (n>1)
		{
			DU=DU+sum(t)/2
		}
		r<-r+n*(n-1)/2
	}
	Dmin=sum(sort(ddist)[1:r])
	Dmax=sum(sort(ddist,decreasing = T)[1:r])
	if(Dmin==Dmax)
		result<-NA
	else
		result<-(DU-Dmin)/(Dmax-Dmin)
	result
}

"""


def c_index(X, labels):
    """Calculate CIndex
    inputs
    -------
    X : (np.ndarray) an (n x m) array where n is the number of examples to cluster
        and m is the feature space of examples
    cluster_labels : (np.array) of cluster labels, each cluster_labels[i]
        related to X[i]
        ideally integer type
    output
    -------
    cindex : (float)"""



    # Total Number of pairs of observations belonging to same cluster
    Nw = calc_Nw(labels)

    # Distances between all pairs of points in dataset
    distances = pdist(X, metric='euclidean')

    # Sum of within-cluster distances
    Sw = calc_sw(X, labels)

    # Sum of Nw smallest distances between all poirs of points
    # Sum of Nw largest distances between all pairs of points
    Smin, Smax = calc_smin_smax(distances, Nw)

    # Calculate CIndex
    cindex = (Sw - Smin) / (Smax - Smin)
    return cindex


def calc_smin_smax(distances, n_incluster_pairs):
    """Calculate Smin and Smax,
    Smax is the Sum of Nw largest distances between all pairs of points
    in the entire data set, and
    Smin is the Sum of Nw smallest distances between all pairs of points
    in the entire data set

    inputs
    -------
    distances : (np.ndarray) of shape [m,] where m is the pairwise distances
        between each point [i,k] being clustered. m is calculated as
        m = n_points * (n_points - 1) / 2 where n_points is the number of points
        in the entire dataset
    n_incluster_pairs : (int) Total number of pairs of observations belonging
        to the same cluster - See calc Nw
    outputs
    -------
    Smin, Smax : (float)
    """
    n_incluster_pairs = int(n_incluster_pairs) # For indexing
    indicies = np.argsort(distances)

    Smin = np.sum(distances[indicies[:n_incluster_pairs]])
    Smax = np.sum(distances[indicies[-n_incluster_pairs:]])
    return Smin, Smax


def calc_sw(X, cluster_labels):
    """Sum of within-cluster distances"""

    labels = np.array(cluster_labels)
    labels_set = set(cluster_labels)
    n_labels = len(labels_set)

    Sw = []
    for label in labels_set:
        # Loop through each cluster and calculate within cluster distance
        pairs = np.where(labels == label)
        pairs_distance = pdist(X[pairs[0]])
        within_cluster_distance = np.sum(pairs_distance, axis=0)
        Sw.append(within_cluster_distance)

    return np.sum(Sw)


def calc_Nw(cluster_labels):
    """Total number of pairs of observations belonging to the same cluster

    N_w = \sum_{k=1}^{q} \frac{n_k (n_k-1)}{2}
    inputs
    -------
    labels : (iterable) of labels"""

    cluster_labels = np.array(cluster_labels)
    labels_set = set(cluster_labels)
    n_labels = len(labels_set)

    Nw = []
    for label in labels_set:
        n_examples = np.sum(np.where(cluster_labels == label, 1, 0))
        n_cluster_pairs = n_examples * (n_examples - 1) / 2 # Combinations
        Nw.append(n_cluster_pairs)

    return int(np.sum(Nw))
