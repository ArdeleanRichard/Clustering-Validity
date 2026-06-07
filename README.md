# The reason why cluster validity indices are wrong 

This repository contains the code and configurations for evaluating Cluster Validity Indices.

<!-- 
[![DOI](https://img.shields.io/badge/DOI-10.3390/diagnostics15141823-blue)](https://doi.org/???) 

This repository contains the code and configurations used in the study titled:
**"The reason why cluster validity indices are wrong"**
by Eugen-Richard Ardelean, Mircea Susca, and Raluca Portase, published in ???
-->

## 📄 Overview

Cluster Validity Indices (CVIs) are widely used to assess clustering quality and to estimate the number of clusters, but the paper shows that many of them fail to assign the best score to the best labels.

The study evaluates 28 CVIs across handcrafted labelsets and clustering outputs, and highlights two main limitations:

- CVIs can prefer the wrong partition when clusters are irregular, overlapping, or imbalanced.
- External CVIs can also be biased by class imbalance, where large classes dominate the score.

To address these issues, the paper proposes:

- **Arboris Distance (AD)**: a minimum-spanning-tree-based distance that captures data topology better than plain Euclidean distance for irregular shapes.
- **AD-based CVIs**: extensions of Silhouette, Davies-Bouldin, and Calinski-Harabasz.
- **AD-IDEA**: a new internal CVI built on Arboris Distance.
- **Balanced external CVIs**: per-class scoring variants designed to reduce imbalance bias.


Project files:
- main_arboris.py : showcasing the Arboris CVIs on a simple case
- main_supp_clustering_comparison.py : showcasing the Arboris distance in clustering on a simple case
- main_supp_external_CVIs.py : showcasing the balanced versions for external CVIs

- main_analysis_count_labelsets.py : simple analysis counting the number of correct evaluations across datasets for the CVIs  (errors = how many handcrafted labels give higher scores than the ground truth labels)
- main_analysis_count_clustering.py : simple analysis counting the number of correct evaluations across datasets for the CVIs (errors = how many clustering labels give higher scores than the ground truth labels)
- main_analysis_count_clustering_stats.py : statistical analysis of best count analysis

- main_clustering_grid_search.py : run a clustering grid search to find the best parametrisation for each datasets of each clustering algorithm
- main_analysis_correlation_clustering_best.py : correlation analysis of the best clustering for each dataset
- main_analysis_correlation_clustering_all.py : correlation analysis of the all clustering configurations for each dataset

---

## 📊 Datasets

The experiments use a mix of synthetic benchmark datasets and real-world datasets.

### Synthetic datasets
The paper evaluates synthetic data with a wide range of behaviors, including:

- varying numbers of clusters
- overlap and increasing overlap
- strong imbalance
- irregular / non-globular shapes
- embedded clusters
- noisy clusters
- outliers
- linearly separable and hard-to-separate patterns

### Real-world datasets
The paper also evaluates 8 UCI datasets.


---

## 🧠 Algorithms Evaluated

The paper compares multiple clustering validation approaches and clustering algorithms.

### Internal CVIs
The study includes traditional and modern internal CVIs such as:

- Silhouette (`S`)
- Davies-Bouldin (`DB`)
- Calinski-Harabasz (`CH`)
- Ball Hall (`BH`)
- Xie-Beni (`XB`)
- Dunn (`D`)
- Hartigan (`H`)
- DBCV, CDbw, VIASCKDE, COP, CS, SF, SD, SDbw, and others

### AD-based CVIs
The paper introduces AD-based variants of well-known CVIs:

- `AD-S`
- `AD-DB`
- `AD-CH`
- `AD-IDEA`

### Balanced external CVIs
The paper also proposes balanced external variants, including:

- Balanced Rand Index
- Balanced Adjusted Rand Index
- class-wise balanced extensions for external CVIs

### Clustering algorithms used in the evaluation
The paper evaluates labelsets and clusterings produced by:

- KMeans
- DBSCAN
- HDBSCAN
- MeanShift
- AgglomerativeClustering
- SpectralClustering



## 📈 Results Summary

The main findings reported in the paper are:

- Many CVIs do **not** assign their best score to the ground-truth labels.
- This failure is especially strong for **irregular cluster shapes**, **imbalance**, and **overlap**.
- Internal CVIs can be useful, but they are often sensitive to the number of clusters and can prefer the wrong partition.
- External CVIs can also fail on imbalanced data, because majority classes dominate the score.
- The proposed **Arboris Distance** improves the behavior of S, DB, and CH on many synthetic benchmarks.
- **AD-IDEA** is the strongest proposed index in the handcrafted-label analysis.
- Performance on real datasets remains data dependent, reinforcing the paper’s conclusion that no single CVI is universally best.

---
<!-- 
## 📜 Citation

If you use this work or code, please cite:

```bibtex
@article{Ardelean2026-Clustering,
}
```

---
-->

## 📬 Contact

For questions, please contact:
📧 [ardeleaneugenrichard@gmail.com](mailto:ardeleaneugenrichard@gmail.com)

---

