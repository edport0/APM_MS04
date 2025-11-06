import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from Mesh_generator import build_mesh_from_points


# ---------------------------------------------------------------------
# Data structure for the hierarchical boxes
# ---------------------------------------------------------------------
@dataclass
class ClusterNode:
    indices: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    children: list
    level: int
    diameter: float = 0.0

    @property
    def is_leaf(self):
        return len(self.children) == 0


# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------
def compute_diameter(node: ClusterNode) -> float:
    """Return box diameter as the diagonal length of the bounding box."""
    diff = node.bbox_max - node.bbox_min
    return np.sqrt(np.sum(diff**2))


def distbox(nodeA: ClusterNode, nodeB: ClusterNode) -> float:
    """Compute the distance between two boxes."""
    dx = max(0.0, max(nodeA.bbox_min[0] - nodeB.bbox_max[0],
                      nodeB.bbox_min[0] - nodeA.bbox_max[0]))
    dy = max(0.0, max(nodeA.bbox_min[1] - nodeB.bbox_max[1],
                      nodeB.bbox_min[1] - nodeA.bbox_max[1]))
    return np.sqrt(dx**2 + dy**2)


# ---------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------
def build_tree(points: np.ndarray, Nleaf=10) -> ClusterNode:
    """
    Build the cluster tree by recursively bisecting along the longest
    box dimension, until each box contains ≤ Nleaf points.
    """
    N = points.shape[0]
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()

    root = ClusterNode(np.arange(N),
                       np.array([xmin, ymin]),
                       np.array([xmax, ymax]),
                       [], 0)
    root.diameter = compute_diameter(root)
    current_level = [root]

    while True:
        next_level = []
        stop = True
        for node in current_level:
            idx = node.indices
            if len(idx) > Nleaf:
                stop = False
                pts = points[idx]
                dx = node.bbox_max[0] - node.bbox_min[0]
                dy = node.bbox_max[1] - node.bbox_min[1]

                # Decide which direction to bisect
                if dx >= dy:
                    split_val = 0.5 * (node.bbox_min[0] + node.bbox_max[0])
                    left_mask = pts[:, 0] <= split_val
                    right_mask = ~left_mask
                    left_bbox_min = np.array([node.bbox_min[0], node.bbox_min[1]])
                    left_bbox_max = np.array([split_val, node.bbox_max[1]])
                    right_bbox_min = np.array([split_val, node.bbox_min[1]])
                    right_bbox_max = np.array([node.bbox_max[0], node.bbox_max[1]])
                else:
                    split_val = 0.5 * (node.bbox_min[1] + node.bbox_max[1])
                    left_mask = pts[:, 1] <= split_val
                    right_mask = ~left_mask
                    left_bbox_min = np.array([node.bbox_min[0], node.bbox_min[1]])
                    left_bbox_max = np.array([node.bbox_max[0], split_val])
                    right_bbox_min = np.array([node.bbox_min[0], split_val])
                    right_bbox_max = np.array([node.bbox_max[0], node.bbox_max[1]])

                if np.all(left_mask) or np.all(right_mask):
                    next_level.append(node)
                    continue

                left_idx = idx[left_mask]
                right_idx = idx[right_mask]

                left_node = ClusterNode(left_idx, left_bbox_min, left_bbox_max, [], node.level + 1)
                right_node = ClusterNode(right_idx, right_bbox_min, right_bbox_max, [], node.level + 1)
                left_node.diameter = compute_diameter(left_node)
                right_node.diameter = compute_diameter(right_node)
                node.children = [left_node, right_node]
                next_level.extend([left_node, right_node])
            else:
                next_level.append(node)

        if stop:
            break
        current_level = next_level

    return root


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
def plot_tree(node, points, ax=None, color='C0'):
    if ax is None:
        fig, ax = plt.subplots()
    x0, y0 = node.bbox_min
    x1, y1 = node.bbox_max
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, lw=0.8)
    for child in node.children:
        plot_tree(child, points, ax, color)
    if node.level == 0:
        ax.plot(points[:, 0], points[:, 1], 'k.', ms=4)
        ax.set_aspect('equal')
        ax.set_title("Cluster tree with geometric bisection")
    return ax


# ---------------------------------------------------------------------
# Example usage (integrates smoothly with Mesh_generator)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example: circular boundary
    R = 1.0
    n_points = 64
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    pts = np.column_stack([R*np.cos(theta), R*np.sin(theta)])
    mesh = build_mesh_from_points(pts)

    # Build tree
    Nleaf = 8
    root = build_tree(mesh.points, Nleaf=Nleaf)
    print("Geometric bisection tree built.")

    # Plot
    plot_tree(root, mesh.points)
    plt.show()

    # Example: compute and print some distances / diameters
    leaves = []

    def collect_leaves(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            for c in node.children:
                collect_leaves(c)

    collect_leaves(root)
    print(f"Number of leaves: {len(leaves)}")
    for i, leaf in enumerate(leaves[:5]):
        print(f"Leaf {i}: diameter = {leaf.diameter:.3f}, n_points = {len(leaf.indices)}")
    if len(leaves) >= 2:
        d = distbox(leaves[0], leaves[1])
        print(f"Example distbox(leaves[0], leaves[1]) = {d:.3f}")
