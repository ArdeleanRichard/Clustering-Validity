# Cluster Validity Indices and Clustering Algorithms
https://github.com/ErfanAmani2000/Cluster-Validity-Indices/tree/main

This repository contains implementations of various clustering algorithms and cluster validity indices, including a novel SE measure. The main purpose of this project is to provide an efficient tool for clustering analysis and validation, with a particular focus on the new SE cluster validity measure.

## Abstract
Due to the absence of labels or so-called target variables, clustering validation, unlike classification, is more complex. Thus, cluster evaluation is challenging both in research projects and applications. While many clustering validity indices are addressed in the literature, most of them, even those widely used in applications, cannot handle arbitrary shapes. This paper proposes a novel clustering validity index called SE that can accommodate arbitrary shapes and is much more powerful in capturing the data's structure. The index is based on the significance of separation, for which a novel separation measure is proposed. In this approach, clusters that are significantly separated from each other take a maximum value of one. This threshold ensures that sparseness does not lose its importance and contribution to calculating the score for the cluster validity index. Extensive experiments are conducted on thirty-three natural and synthetic datasets, revealing the promising efficacy of the proposed algorithm. Contrary to some existing density-based clustering validity indices, the computational expensiveness test, again performed on the thirty-three datasets mentioned above, demonstrates the proposed index's applicability even on large-scale datasets.

The corresponding paper, titled **"Density-based cluster validity index using a separation measure"**, is currently under review after the first revision in the *Neurocomputing* journal.

## Features

- **Clustering Algorithms**:
  - K-Means Clustering
  - Agglomerative Clustering
  - OPTICS Clustering
  - DBSCAN Clustering
  - RSC Algorithm (Novel Clustering Algorithm)

- **Cluster Validity Indices**:
  - **Davies-Bouldin Score** ([Davies & Bouldin, 1979](https://doi.org/10.1109/TPAMI.1979.4766909))
  - **Silhouette Score** ([Rousseeuw, 1987](https://doi.org/10.1016/0377-0427(87)90125-7))
  - **S_Dbw Index** ([Halkidi & Vazirgiannis, 2001](https://doi.org/10.1007/3-540-47745-1_36))
  - **CDbw Index** ([Chen et al., 2002](https://doi.org/10.1109/ICPR.2002.1047902))
  - **LCCV Index** ([Zhang, 2011](https://doi.org/10.1016/j.patcog.2011.06.002))
  - **DBCV Index** ([Moulavi et al., 2014](https://doi.org/10.1007/978-3-319-06605-9_1))
  - **NCCV Index** ([Wiroonsri, 2024](https://doi.org/10.1016/j.patcog.2023.109910))
  - **SE Measure** (Novel Cluster Validity Measure introduced in the referenced paper)

## Installation

To use the code in this repository, clone it to your local machine and install the necessary dependencies.

```sh
git clone https://github.com/ErfanAmani2000/Cluster-Validity-Indices.git
cd cluster-validity-indices
