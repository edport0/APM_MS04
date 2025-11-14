import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from Mesh_generator import build_mesh_from_points


# ---------------------------------------------------------------------
# Cluster node definition
# ---------------------------------------------------------------------
@dataclass
class ClusterNode:
    indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    children: list
    level: int

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def compute_bbox(points):
    """Return bounding box of given point cloud."""
    return np.min(points, axis=0), np.max(points, axis=0)


def compute_diameter(node: ClusterNode):
    diff = node.bbox_max - node.bbox_min
    return np.sqrt(np.sum(diff**2))


# ---------------------------------------------------------------------
# Balanced tree construction (median split)
# ---------------------------------------------------------------------
def build_balanced_tree(points: np.ndarray, Nleaf: int = 10) -> ClusterNode:
    """
    Build a binary tree by recursively bisecting along the largest
    dimension at the median coordinate (balanced subdivision).
    """
    N = len(points)
    bbox_min, bbox_max = compute_bbox(points)
    root = ClusterNode(np.arange(N), bbox_min, bbox_max, [], level=0)

    def _subdivide(node: ClusterNode):
        idx = node.indices
        pts = points[idx]
        if len(idx) <= Nleaf:
            return

        # Find longest axis
        lengths = node.bbox_max - node.bbox_min
        axis = np.argmax(lengths)

        # Split by median along that axis
        median_val = np.median(pts[:, axis])
        left_mask = pts[:, axis] <= median_val
        right_mask = ~left_mask

        # Safety guard if all points fall on one side
        if np.all(left_mask) or np.all(right_mask):
            return

        left_idx = idx[left_mask]
        right_idx = idx[right_mask]

        left_bbox_min, left_bbox_max = compute_bbox(points[left_idx])
        right_bbox_min, right_bbox_max = compute_bbox(points[right_idx])

        left_node = ClusterNode(left_idx, left_bbox_min, left_bbox_max, [], node.level + 1)
        right_node = ClusterNode(right_idx, right_bbox_min, right_bbox_max, [], node.level + 1)
        node.children = [left_node, right_node]

        # Recurse
        _subdivide(left_node)
        _subdivide(right_node)

    _subdivide(root)
    return root


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def plot_tree(node: ClusterNode, points, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    x0, y0 = node.bbox_min
    x1, y1 = node.bbox_max
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'b-', lw=0.8)
    for c in node.children:
        plot_tree(c, points, ax)
    if node.level == 0:
        ax.plot(points[:, 0], points[:, 1], 'k.', ms=4)
        ax.set_aspect('equal')
        ax.set_title("Balanced tree (median split per longest axis)")
    return ax


# ---------------------------------------------------------------------
# Example (as in the previous questions)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    R = 1.0
    n_points = 64
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    #pts = np.column_stack([R*np.cos(theta), R*np.sin(theta)])
    pts = np.column_stack([np.linspace(0, R, n_points, endpoint = True), np.zeros(n_points)])
    mesh = build_mesh_from_points(pts)

    Nleaf = 8
    root = build_balanced_tree(mesh.points, Nleaf=Nleaf)

    # Count levels
    def count_levels(node):
        if node.is_leaf:
            return node.level
        return max(count_levels(c) for c in node.children)
    n_levels = count_levels(root)
    print(f"Balanced tree built with {n_levels} levels (Nleaf={Nleaf})")

    plot_tree(root, mesh.points)
    plt.show()
