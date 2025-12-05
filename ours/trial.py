import numpy as np
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import MinMaxScaler

from load_datasets import create_data1, create_data2
from load_labelsets import assign_labels_by_given_line, vertical_line, diagonal_line, horizontal_line

import numpy as np
from collections import defaultdict, deque
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from sklearn.preprocessing import MinMaxScaler


def centroid_id_from_data_fast(data):
    # Compute all pairwise squared distances at once
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    pairwise_sq = np.sum(diff ** 2, axis=2)
    sum_sq = np.sum(pairwise_sq, axis=1)
    min_idx = np.argmin(sum_sq)

    return data[min_idx]



class AdaptiveGridCell:
    """Represents a cell in the adaptive grid."""

    def __init__(self, bounds, points, indices, depth=0, n_total_points=None):
        """
        Parameters:
        - bounds: tuple of (min_coords, max_coords)
        - points: ndarray of points in this cell
        - indices: original indices of points
        - depth: recursion depth
        - n_total_points: total points in dataset for relative density
        """
        self.bounds = bounds
        self.points = points
        self.indices = indices
        self.depth = depth

        # Compute cell properties
        self.n_points = len(points)

        # Compute volume based on actual point spread
        if len(points) > 0:
            point_mins = np.min(points, axis=0)
            point_maxs = np.max(points, axis=0)
            point_ranges = point_maxs - point_mins
            # Add small epsilon to avoid zero volume
            point_ranges = np.maximum(point_ranges, 1e-6)
            self.volume = np.prod(point_ranges)
        else:
            self.volume = np.prod(bounds[1] - bounds[0])

        # Relative density: (n_points / n_total) / volume
        if n_total_points is not None and n_total_points > 0:
            self.density = (self.n_points / n_total_points) / self.volume
        else:
            self.density = self.n_points / self.volume if self.volume > 0 else 0

        # self.centroid = np.mean(points, axis=0) if len(points) > 0 else (bounds[0] + bounds[1]) / 2
        self.centroid = centroid_id_from_data_fast(points) if len(points) > 0 else (bounds[0] + bounds[1]) / 2
        self.bbox_center = (bounds[0] + bounds[1]) / 2  # Center of bounding box

        self.children = None
        self.is_leaf = True
        self.cell_id = None


class AdaptiveGridMST:
    """Adaptive grid-based MST for cluster validity."""

    def __init__(self, data, labels=None, min_points=5, max_depth=10, density_threshold=None):
        """
        Parameters:
        - data: ndarray, shape (n_samples, n_features)
        - labels: cluster labels (optional, for visualization)
        - min_points: minimum points per cell before stopping split
        - max_depth: maximum recursion depth
        - density_threshold: if provided, split cells with density above this
        """
        self.data = data
        self.labels = labels
        self.n_samples, self.n_dims = data.shape
        self.min_points = min_points
        self.max_depth = max_depth

        # Compute density threshold as median density if not provided
        if density_threshold is None:
            data_bounds = (np.min(data, axis=0), np.max(data, axis=0))
            data_ranges = data_bounds[1] - data_bounds[0]
            total_volume = np.prod(data_ranges)
            # Relative density threshold
            self.density_threshold = (1.0 / self.n_samples) / total_volume * 0.5
        else:
            self.density_threshold = density_threshold

        # Build adaptive grid
        self.root = None
        self.leaf_cells = []
        self._build_grid()

        # Build graph from cells
        self.cell_graph = None
        self.cell_mst = None
        self._build_cell_graph()
        self._build_mst()

    def _build_grid(self):
        """Build adaptive KD-tree-like grid."""
        bounds = (np.min(self.data, axis=0), np.max(self.data, axis=0))
        all_indices = np.arange(self.n_samples)

        self.root = AdaptiveGridCell(bounds, self.data, all_indices, depth=0,
                                     n_total_points=self.n_samples)
        self._split_cell(self.root, axis=0)

        # Assign IDs to leaf cells
        for i, cell in enumerate(self.leaf_cells):
            cell.cell_id = i

        # print(f"Created {len(self.leaf_cells)} leaf cells")

    def _split_cell(self, cell, axis):
        """Recursively split cell based on density."""
        # Stopping conditions
        if (cell.depth >= self.max_depth or
                cell.n_points <= self.min_points or
                cell.density < self.density_threshold):
            self.leaf_cells.append(cell)
            return

        # Split along axis at median
        points = cell.points
        if len(points) < 2:
            self.leaf_cells.append(cell)
            return

        split_value = np.median(points[:, axis])

        # Create masks for left and right
        left_mask = points[:, axis] <= split_value
        right_mask = ~left_mask

        # If split doesn't separate points, stop
        if not np.any(left_mask) or not np.any(right_mask):
            self.leaf_cells.append(cell)
            return

        # Create child cells
        left_bounds = (cell.bounds[0].copy(), cell.bounds[1].copy())
        left_bounds[1][axis] = split_value

        right_bounds = (cell.bounds[0].copy(), cell.bounds[1].copy())
        right_bounds[0][axis] = split_value

        left_cell = AdaptiveGridCell(
            left_bounds,
            points[left_mask],
            cell.indices[left_mask],
            cell.depth + 1,
            n_total_points=self.n_samples
        )

        right_cell = AdaptiveGridCell(
            right_bounds,
            points[right_mask],
            cell.indices[right_mask],
            cell.depth + 1,
            n_total_points=self.n_samples
        )

        cell.children = (left_cell, right_cell)
        cell.is_leaf = False

        # Recursively split children
        next_axis = (axis + 1) % self.n_dims
        self._split_cell(left_cell, next_axis)
        self._split_cell(right_cell, next_axis)

    def _cells_are_adjacent(self, cell1, cell2):
        """Check if two cells share a face/edge (are adjacent)."""
        bounds1 = cell1.bounds
        bounds2 = cell2.bounds

        touch_count = 0
        overlap_count = 0

        for dim in range(self.n_dims):
            min1, max1 = bounds1[0][dim], bounds1[1][dim]
            min2, max2 = bounds2[0][dim], bounds2[1][dim]

            # Check if they touch (share boundary)
            if np.isclose(max1, min2) or np.isclose(max2, min1):
                touch_count += 1
            # Check if they overlap in this dimension
            elif max1 > min2 and max2 > min1:
                overlap_count += 1

        # Adjacent if they touch in at least one dimension and overlap in others
        return touch_count >= 1 and (touch_count + overlap_count) == self.n_dims

    def _compute_edge_weight(self, cell1, cell2):
        """
        Compute edge weight between two adjacent cells.

        Formula: euclidean_distance / geometric_mean(density1, density2)

        Intuition:
        - Larger distance = higher weight (cells far apart)
        - Higher density = lower weight (dense regions should be connected)
        - Geometric mean prevents one very high density from dominating
        """
        # Euclidean distance between bounding box centers
        euclidean_dist = np.linalg.norm(cell1.centroid - cell2.centroid)

        # Geometric mean of densities (avoids issues with zero density)
        eps = 1e-10
        geometric_mean_density = np.sqrt(max(cell1.density, eps) * max(cell2.density, eps))

        # Weight: distance divided by density
        # High density regions -> low weight (should be connected)
        # Low density regions -> high weight (boundaries)
        weight = euclidean_dist # / geometric_mean_density

        return weight

    def _build_cell_graph(self):
        """Build graph where nodes are cells and edges connect adjacent cells."""
        n_cells = len(self.leaf_cells)
        self.cell_graph = defaultdict(list)

        # Find all adjacent pairs
        edges = []
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                if self._cells_are_adjacent(self.leaf_cells[i], self.leaf_cells[j]):
                    weight = self._compute_edge_weight(self.leaf_cells[i], self.leaf_cells[j])
                    edges.append((i, j, weight))
                    self.cell_graph[i].append((j, weight))
                    self.cell_graph[j].append((i, weight))

        # print(f"Created cell graph with {len(edges)} edges")
        return edges

    def _build_mst(self):
        """Build MST from cell graph using Prim's algorithm."""
        n_cells = len(self.leaf_cells)
        if n_cells <= 1:
            self.cell_mst = []
            return

        visited = np.zeros(n_cells, dtype=bool)
        mst_edges = []
        pq = []

        # Start from cell 0
        visited[0] = True
        for neighbor, weight in self.cell_graph[0]:
            heapq.heappush(pq, (weight, 0, neighbor))

        while len(mst_edges) < n_cells - 1 and pq:
            weight, u, v = heapq.heappop(pq)

            if visited[v]:
                continue

            mst_edges.append((u, v, weight))
            visited[v] = True

            for neighbor, edge_weight in self.cell_graph[v]:
                if not visited[neighbor]:
                    heapq.heappush(pq, (edge_weight, v, neighbor))

        self.cell_mst = mst_edges
        # print(f"Built MST with {len(mst_edges)} edges")

    def _compute_cell_purity(self, cell):
        """
        Compute purity of a cell (1.0 = all same label, 0.0 = completely mixed).
        """
        if cell.n_points == 0 or self.labels is None:
            return 1.0

        cell_labels = self.labels[cell.indices]
        majority_count = np.bincount(cell_labels).max()
        purity = majority_count / cell.n_points

        return purity

    def _compute_cell_impurity_penalty(self, cell):
        """
        Compute impurity penalty (0.0 = pure, higher = more mixed).
        Using entropy-like measure.
        """
        if cell.n_points == 0 or self.labels is None:
            return 0.0

        cell_labels = self.labels[cell.indices]
        label_counts = np.bincount(cell_labels)
        label_probs = label_counts / cell.n_points

        # Entropy: -sum(p * log(p))
        # Higher entropy = more mixed
        entropy = -np.sum([p * np.log2(p) for p in label_probs if p > 0])

        return entropy

    def visualize_grid(self, filename='grid_structure.png'):
        """Visualize the adaptive grid structure."""
        if self.n_dims != 2:
            print("Visualization only supported for 2D data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Grid with data points
        ax = axes[0]
        if self.labels is not None:
            scatter = ax.scatter(self.data[:, 0], self.data[:, 1],
                                 c=self.labels, cmap='tab10', s=10, alpha=0.6)
        else:
            scatter = ax.scatter(self.data[:, 0], self.data[:, 1],
                                 s=10, alpha=0.6, c='blue')

        # Draw cell boundaries
        for cell in self.leaf_cells:
            bounds = cell.bounds
            rect = patches.Rectangle(
                (bounds[0][0], bounds[0][1]),
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                linewidth=1, edgecolor='black', facecolor='none', alpha=0.5
            )
            ax.add_patch(rect)

        ax.set_title(f'Adaptive Grid ({len(self.leaf_cells)} cells)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Plot 2: Grid colored by density
        ax = axes[1]
        densities = [cell.density for cell in self.leaf_cells]
        max_density = max(densities) if densities else 1

        for cell in self.leaf_cells:
            bounds = cell.bounds
            color_intensity = cell.density / max_density
            rect = patches.Rectangle(
                (bounds[0][0], bounds[0][1]),
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                linewidth=0.5, edgecolor='black',
                facecolor=plt.cm.YlOrRd(color_intensity), alpha=0.7
            )
            ax.add_patch(rect)

        ax.set_title('Grid Colored by Relative Density')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xlim(self.data[:, 0].min(), self.data[:, 0].max())
        ax.set_ylim(self.data[:, 1].min(), self.data[:, 1].max())

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved grid visualization to {filename}")
        plt.close()

    def visualize_cell_graph(self, filename='cell_graph.png'):
        """Visualize the cell graph."""
        if self.n_dims != 2:
            print("Visualization only supported for 2D data")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw cells
        for cell in self.leaf_cells:
            bounds = cell.bounds
            rect = patches.Rectangle(
                (bounds[0][0], bounds[0][1]),
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                linewidth=0.5, edgecolor='gray', facecolor='lightblue', alpha=0.3
            )
            ax.add_patch(rect)

            # Draw bbox center
            ax.plot(cell.bbox_center[0], cell.bbox_center[1], 'ro', markersize=5)

        # Draw edges
        for i, neighbors in self.cell_graph.items():
            cell1 = self.leaf_cells[i]
            for j, weight in neighbors:
                if i < j:  # Draw each edge once
                    cell2 = self.leaf_cells[j]
                    ax.plot([cell1.bbox_center[0], cell2.bbox_center[0]],
                            [cell1.bbox_center[1], cell2.bbox_center[1]],
                            'b-', alpha=0.3, linewidth=0.5)

        ax.set_title(f'Cell Graph ({len(self.leaf_cells)} nodes, {sum(len(v) for v in self.cell_graph.values()) // 2} edges)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved cell graph to {filename}")
        plt.close()

    def visualize_mst(self, filename='cell_mst.png'):
        """Visualize the MST of cells."""
        if self.n_dims != 2:
            print("Visualization only supported for 2D data")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: MST with data points
        ax = axes[0]
        if self.labels is not None:
            ax.scatter(self.data[:, 0], self.data[:, 1],
                       c=self.labels, cmap='tab10', s=10, alpha=0.5)
        else:
            ax.scatter(self.data[:, 0], self.data[:, 1],
                       s=10, alpha=0.5, c='lightgray')

        # Draw cell boundaries
        for cell in self.leaf_cells:
            bounds = cell.bounds
            rect = patches.Rectangle(
                (bounds[0][0], bounds[0][1]),
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.3
            )
            ax.add_patch(rect)

        # Draw bbox centers
        for cell in self.leaf_cells:
            # ax.plot(cell.bbox_center[0], cell.bbox_center[1], 'ko', markersize=4)
            ax.plot(cell.centroid[0], cell.centroid[1], 'ko', markersize=4)

        # Draw MST edges with color based on weight
        if self.cell_mst:
            weights = [w for _, _, w in self.cell_mst]
            max_weight = max(weights)

            for u, v, weight in self.cell_mst:
                cell1 = self.leaf_cells[u]
                cell2 = self.leaf_cells[v]
                color_intensity = weight / max_weight
                # ax.plot([cell1.bbox_center[0], cell2.bbox_center[0]],
                #         [cell1.bbox_center[1], cell2.bbox_center[1]],
                #         'r-', alpha=0.7, linewidth=2,
                #         color=plt.cm.Reds(color_intensity))
                ax.plot([cell1.centroid[0], cell2.centroid[0]],
                        [cell1.centroid[1], cell2.centroid[1]],
                        'r-', alpha=0.7, linewidth=2,
                        color=plt.cm.Reds(color_intensity))

        ax.set_title('Cell MST (edge color = weight)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Plot 2: MST only with labels and purity
        ax = axes[1]

        # Draw cells colored by cluster majority, alpha by purity
        if self.labels is not None:
            for cell in self.leaf_cells:
                if cell.n_points > 0:
                    cell_labels = self.labels[cell.indices]
                    majority_label = np.bincount(cell_labels).argmax()
                    purity = self._compute_cell_purity(cell)
                    bounds = cell.bounds

                    # Alpha represents purity (1.0 = pure, lower = mixed)
                    rect = patches.Rectangle(
                        (bounds[0][0], bounds[0][1]),
                        bounds[1][0] - bounds[0][0],
                        bounds[1][1] - bounds[0][1],
                        linewidth=0.5, edgecolor='black',
                        facecolor=plt.cm.tab10(majority_label % 10),
                        alpha=purity * 0.7 + 0.3  # Map purity to alpha
                    )
                    ax.add_patch(rect)

        # Draw MST edges
        if self.cell_mst:
            for u, v, weight in self.cell_mst:
                cell1 = self.leaf_cells[u]
                cell2 = self.leaf_cells[v]
                # ax.plot([cell1.bbox_center[0], cell2.bbox_center[0]],
                #         [cell1.bbox_center[1], cell2.bbox_center[1]],
                #         'k-', alpha=0.8, linewidth=2)
                ax.plot([cell1.centroid[0], cell2.centroid[0]],
                        [cell1.centroid[1], cell2.centroid[1]],
                        'k-', alpha=0.8, linewidth=2)

        # Draw bbox centers
        for cell in self.leaf_cells:
            # ax.plot(cell.bbox_center[0], cell.bbox_center[1], 'wo', markersize=5, markeredgecolor='black', markeredgewidth=1)
            ax.plot(cell.centroid[0], cell.centroid[1], 'wo', markersize=5, markeredgecolor='black', markeredgewidth=1)

        ax.set_title('Cell MST (transparency = purity)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved MST visualization to {filename}")
        plt.close()

    def compute_cluster_validity_score(self):
        """
        Compute cluster validity score incorporating cell purity.

        Score components:
        1. Separation: ratio of inter-cluster to intra-cluster edge weights
        2. Purity penalty: average cell impurity (entropy)

        Final score: separation_ratio * (1 - normalized_impurity)
        Higher is better.
        """
        if self.labels is None:
            # print("Labels required for cluster validity")
            return None

        if not self.cell_mst:
            return 0.0

        # 1. Classify MST edges as intra-cluster or inter-cluster
        intra_weights = []
        inter_weights = []

        for u, v, weight in self.cell_mst:
            cell1 = self.leaf_cells[u]
            cell2 = self.leaf_cells[v]

            if cell1.n_points > 0 and cell2.n_points > 0:
                # Get majority label for each cell
                # label1 = np.bincount(self.labels[cell1.indices]).argmax()
                # label2 = np.bincount(self.labels[cell2.indices]).argmax()

                # if label1 == label2:
                if (len(np.setdiff1d(np.unique(self.labels[cell1.indices]), np.unique(self.labels[cell2.indices]))) == 0 and
                        len(np.setdiff1d(np.unique(self.labels[cell2.indices]), np.unique(self.labels[cell1.indices]))) == 0):
                    intra_weights.append(weight)
                else:
                    inter_weights.append(weight)

        if not intra_weights or not inter_weights:
            separation_ratio = 0.0
        else:
            # Separation: ratio of mean inter to mean intra
            min_inter = np.min(inter_weights)
            max_intra = np.max(intra_weights)
            separation_ratio = min_inter / (max_intra + 1e-10)

        # 2. Compute average cell impurity (entropy-based)
        impurities = []
        for cell in self.leaf_cells:
            if cell.n_points > 0:
                impurity = self._compute_cell_impurity_penalty(cell)
                impurities.append(impurity)

        avg_impurity = np.mean(impurities) if impurities else 0.0

        # Normalize impurity by maximum possible entropy
        # For k clusters, max entropy = log2(k)
        n_clusters = len(np.unique(self.labels))
        max_entropy = np.log2(n_clusters) if n_clusters > 1 else 1.0
        normalized_impurity = avg_impurity / max_entropy if max_entropy > 0 else 0.0

        # 3. Compute average purity for reporting
        purities = []
        for cell in self.leaf_cells:
            if cell.n_points > 0:
                purity = self._compute_cell_purity(cell)
                purities.append(purity)
        avg_purity = np.mean(purities) if purities else 0.0

        # Final score: separation * purity_factor
        # Higher separation and higher purity = better score
        purity_factor = 1.0 - normalized_impurity
        final_score = separation_ratio * purity_factor

        return {
            'final_score': final_score,
            'separation_ratio': separation_ratio,
            'avg_purity': avg_purity,
            'avg_impurity': avg_impurity,
            'normalized_impurity': normalized_impurity,
            'purity_factor': purity_factor
        }

    def analyze_clustering(self):
        """Provide detailed analysis of clustering quality."""
        if self.labels is None:
            print("Labels required for analysis")
            return

        print("\n" + "=" * 60)
        print("CLUSTER ANALYSIS")
        print("=" * 60)

        # Cell statistics
        print(f"\nGrid Statistics:")
        print(f"  Total cells: {len(self.leaf_cells)}")
        print(f"  Points per cell: {self.n_samples / len(self.leaf_cells):.2f} (avg)")

        densities = [cell.density for cell in self.leaf_cells]
        print(f"  Relative density range: [{min(densities):.6f}, {max(densities):.6f}]")
        print(f"  Relative density mean: {np.mean(densities):.6f}")

        volumes = [cell.volume for cell in self.leaf_cells]
        print(f"  Cell volume range: [{min(volumes):.6f}, {max(volumes):.6f}]")

        # Cluster purity per cell
        purities = []
        impurities = []
        for cell in self.leaf_cells:
            if cell.n_points > 0:
                purity = self._compute_cell_purity(cell)
                impurity = self._compute_cell_impurity_penalty(cell)
                purities.append(purity)
                impurities.append(impurity)

        print(f"\nCell Purity Analysis:")
        print(f"  Mean purity: {np.mean(purities):.4f}")
        print(f"  Median purity: {np.median(purities):.4f}")
        print(f"  Cells with 100% purity: {sum(1 for p in purities if p == 1.0)}/{len(purities)}")
        print(f"  Cells with <50% purity: {sum(1 for p in purities if p < 0.5)}/{len(purities)}")

        print(f"\nCell Impurity (Entropy) Analysis:")
        print(f"  Mean impurity: {np.mean(impurities):.4f}")
        print(f"  Max possible entropy: {np.log2(len(np.unique(self.labels))):.4f}")

        # MST edge analysis
        if self.cell_mst:
            intra_weights = []
            inter_weights = []

            for u, v, weight in self.cell_mst:
                cell1 = self.leaf_cells[u]
                cell2 = self.leaf_cells[v]

                if cell1.n_points > 0 and cell2.n_points > 0:
                    label1 = np.bincount(self.labels[cell1.indices]).argmax()
                    label2 = np.bincount(self.labels[cell2.indices]).argmax()

                    if label1 == label2:
                        intra_weights.append(weight)
                    else:
                        inter_weights.append(weight)

            print(f"\nMST Edge Analysis:")
            print(f"  Total MST edges: {len(self.cell_mst)}")
            print(f"  Intra-cluster edges: {len(intra_weights)}")
            print(f"  Inter-cluster edges: {len(inter_weights)}")

            if intra_weights:
                print(f"\n  Intra-cluster edge weights:")
                print(f"    Mean: {np.mean(intra_weights):.4f}")
                print(f"    Std: {np.std(intra_weights):.4f}")
                print(f"    Min: {np.min(intra_weights):.4f}")
                print(f"    Max: {np.max(intra_weights):.4f}")

            if inter_weights:
                print(f"\n  Inter-cluster edge weights:")
                print(f"    Mean: {np.mean(inter_weights):.4f}")
                print(f"    Std: {np.std(inter_weights):.4f}")
                print(f"    Min: {np.min(inter_weights):.4f}")
                print(f"    Max: {np.max(inter_weights):.4f}")

            result = self.compute_cluster_validity_score()
            print(f"\n" + "=" * 60)
            print("VALIDITY SCORE BREAKDOWN:")
            print("=" * 60)
            print(f"  Separation Ratio (inter/intra): {result['separation_ratio']:.4f}")
            print(f"  Average Cell Purity: {result['avg_purity']:.4f}")
            print(f"  Average Cell Impurity (entropy): {result['avg_impurity']:.4f}")
            print(f"  Normalized Impurity: {result['normalized_impurity']:.4f}")
            print(f"  Purity Factor (1 - norm_impurity): {result['purity_factor']:.4f}")
            print(f"\n  FINAL SCORE: {result['final_score']:.4f}")
            print(f"    (Higher is better)")





def demo_adaptive_grid_mst():
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

    # Create synthetic clustered data
    # X, labels = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=0.6, random_state=42)
    # each element is a set of datasets



    X, gt = create_data2(1000)
    scale = (-1, 1)
    label_sets = {"gt": gt}
    label_sets = load_labelsets(X, gt, scale, label_sets, list_labelsets=["dfl", "dsl", "vl", "hl", "rl"])
    X = MinMaxScaler(scale).fit_transform(X)

    print("\nBuilding Adaptive Grid MST...")
    grid_mst = AdaptiveGridMST(X, gt, min_points=8, max_depth=8)

    print("\nGenerating visualizations...")
    grid_mst.visualize_grid('01_adaptive_grid.png')
    grid_mst.visualize_cell_graph('02_cell_graph.png')
    grid_mst.visualize_mst('03_cell_mst.png')

    # Test with different clustering qualities
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT CLUSTERINGS")
    print("=" * 60)

    for label_name, label_set in label_sets.items():
        # Good clustering (original)
        print(f"\nX. CLUSTERING ({label_name}):")
        grid_mst = AdaptiveGridMST(X, label_set, min_points=8, max_depth=8)
        score = grid_mst.compute_cluster_validity_score()['final_score']
        print(f"   Validity Score: {score:.4f}")

        grid_mst.analyze_clustering()
        grid_mst.visualize_mst(f'04_{label_name}_clustering_mst.png')


if __name__ == "__main__":
    demo_adaptive_grid_mst()