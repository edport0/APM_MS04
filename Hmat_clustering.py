import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from Mesh_generator import SimpleBoundaryMesh, build_mesh_from_points


@dataclass
class ClusterNode:
    """Node of the geometric binary cluster tree."""
    indices: np.ndarray              # global point indices in this box
    bbox: np.ndarray                 # shape (2, 2): [[xmin, xmax], [ymin, ymax]]
    level: int                       # depth level in the tree (root = 0)
    left: Optional["ClusterNode"] = None
    right: Optional["ClusterNode"] = None

    @property
    def is_leaf(self) -> bool:
        return (self.left is None) and (self.right is None)

    @property
    def n_points(self) -> int:
        return self.indices.size

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bbox[:, 0] + self.bbox[:, 1])

    @property
    def extent(self) -> np.ndarray:
        """Box side lengths in each direction."""
        return self.bbox[:, 1] - self.bbox[:, 0]


def compute_bbox(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Compute axis-aligned bounding box of a subset of points.
    Returns array of shape (2, 2): [[xmin, xmax], [ymin, ymax]].
    """
    pts = points[indices]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return np.array([[xmin, xmax],
                     [ymin, ymax]], dtype=float)


def build_geometric_cluster_tree(
    points: np.ndarray,
    Nleaf: int,
    indices: Optional[np.ndarray] = None,
    level: int = 0,
) -> ClusterNode:
    """
    Build the binary cluster tree using the geometric criterion:
    split the bounding box along the longest axis at its midpoint.

    Parameters
    ----------
    points : (N, 2) array
        Coordinates of all DOFs.
    Nleaf : int
        Maximum number of points per leaf.
    indices : 1D array of ints, optional
        Indices of points in this node (defaults to all points at root).
    level : int
        Current depth in the tree (root = 0).

    Returns
    -------
    ClusterNode
        Root of the (sub)tree.
    """
    if indices is None:
        indices = np.arange(points.shape[0], dtype=int)

    bbox = compute_bbox(points, indices)
    node = ClusterNode(indices=indices, bbox=bbox, level=level)

    # Stopping criterion: small box -> leaf
    if indices.size <= Nleaf:
        return node

    # Choose longest axis
    extent = node.extent
    axis = int(np.argmax(extent))

    # If no extent along this axis (degenerate box), stop splitting
    if extent[axis] <= 0.0:
        return node

    # Split at midpoint of bounding box along this axis
    mid = 0.5 * (bbox[axis, 0] + bbox[axis, 1])
    coords = points[indices, axis]

    left_mask = coords <= mid
    right_mask = ~left_mask

    # Safety: if one side is empty, do not split further (rare in practice)
    if not np.any(left_mask) or not np.any(right_mask):
        return node

    left_indices = indices[left_mask]
    right_indices = indices[right_mask]

    node.left = build_geometric_cluster_tree(
        points, Nleaf, indices=left_indices, level=level + 1
    )
    node.right = build_geometric_cluster_tree(
        points, Nleaf, indices=right_indices, level=level + 1
    )

    return node


# ---- Utilities for analysis / plotting ------------------------------------ #

def tree_depth(root: ClusterNode) -> int:
    """Maximum depth (number of levels) of the tree."""
    if root.is_leaf:
        return root.level
    depths = []
    if root.left is not None:
        depths.append(tree_depth(root.left))
    if root.right is not None:
        depths.append(tree_depth(root.right))
    return max(depths)


def collect_leaves(root: ClusterNode) -> List[ClusterNode]:
    """Return all leaf nodes in depth-first order."""
    leaves: List[ClusterNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf:
            leaves.append(node)
        else:
            # depth-first: process right after left so that left is visited first
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)
    return leaves


def permutation_from_tree(root: ClusterNode) -> np.ndarray:
    """
    Build a permutation of DOF indices by traversing leaves in depth-first order.
    This tends to make spatially close DOFs have consecutive indices,
    as requested in the TP. :contentReference[oaicite:1]{index=1}
    """
    leaves = collect_leaves(root)
    new_order: List[int] = []
    for leaf in leaves:
        # keep the local ordering inside each leaf
        new_order.extend(leaf.indices.tolist())
    return np.array(new_order, dtype=int)


def plot_boxes(root: ClusterNode, ax, **kwargs) -> None:
    """
    Recursively draw all bounding boxes of the tree.
    Useful to visually check the binary partitioning.
    """
    import matplotlib.patches as patches

    stack = [root]
    while stack:
        node = stack.pop()
        bbox = node.bbox
        x0, x1 = bbox[0]
        y0, y1 = bbox[1]
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, **kwargs
        )
        ax.add_patch(rect)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)


def _assign_tree_coordinates(root: ClusterNode) -> None:
    """
    Assign (x, y) coordinates to all nodes of the cluster tree for plotting.
    - y = -level
    - x: leaves numbered from left to right in DFS order,
      internal nodes centered above their children.

    Coordinates are stored as attributes node._x and node._y.
    """
    x_counter = 0

    def dfs(node: ClusterNode):
        nonlocal x_counter
        if node.is_leaf:
            node._x = float(x_counter)
            node._y = -float(node.level)
            x_counter += 1
        else:
            if node.left is not None:
                dfs(node.left)
            if node.right is not None:
                dfs(node.right)
            # center above children
            xs = []
            if node.left is not None:
                xs.append(node.left._x)
            if node.right is not None:
                xs.append(node.right._x)
            node._x = sum(xs) / len(xs)
            node._y = -float(node.level)

    dfs(root)


def plot_cluster_tree(root: ClusterNode, ax=None,
                      show_npoints: bool = True,
                      show_levels: bool = True):
    """
    Plot the binary cluster tree (combinatorial structure).

    Parameters
    ----------
    root : ClusterNode
        Root of the cluster tree.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. If None, a new figure is created.
    show_npoints : bool
        If True, annotate leaf nodes with the number of DOFs.
    show_levels : bool
        If True, label the y-axis with tree levels.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    # assign positions
    _assign_tree_coordinates(root)

    # collect nodes
    nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    # plot edges
    for node in nodes:
        if node.left is not None:
            ax.plot([node._x, node.left._x],
                    [node._y, node.left._y], "-")
        if node.right is not None:
            ax.plot([node._x, node.right._x],
                    [node._y, node.right._y], "-")

    # plot nodes (internal vs leaves)
    xs_leaf, ys_leaf = [], []
    xs_int, ys_int = [], []
    for node in nodes:
        if node.is_leaf:
            xs_leaf.append(node._x)
            ys_leaf.append(node._y)
        else:
            xs_int.append(node._x)
            ys_int.append(node._y)

    if xs_int:
        ax.scatter(xs_int, ys_int, marker="o", s=40, label="internal")
    if xs_leaf:
        ax.scatter(xs_leaf, ys_leaf, marker="s", s=50, label="leaf")

    # optional annotations
    if show_npoints:
        for node in nodes:
            if node.is_leaf:
                ax.text(node._x, node._y - 0.1,
                        f"{node.n_points}",
                        ha="center", va="top", fontsize=8)

    # axis cosmetics
    max_depth = tree_depth(root)
    if show_levels:
        ax.set_yticks([-l for l in range(0, max_depth + 1)])
        ax.set_yticklabels([f"level {l}" for l in range(0, max_depth + 1)])
    else:
        ax.set_yticks([])

    ax.set_xticks([])
    #ax.invert_yaxis()  # root at the top visually

    ax.set_xlabel("clusters (DFS order)")
    ax.set_title("Binary cluster tree")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)

    return ax

def compute_diameter(node: ClusterNode) -> float:
    """
    Return box diameter as the diagonal length of the bounding box.

    For bbox = [[xmin, xmax],
                [ymin, ymax]],
    we use sqrt( (xmax-xmin)^2 + (ymax-ymin)^2 ).
    """
    bbox = node.bbox      # shape (2, 2)
    diff = bbox[:, 1] - bbox[:, 0]
    return float(np.sqrt(np.sum(diff**2)))

def precompute_diameters(root: ClusterNode) -> None:
    """Compute and store the diameter of every node in the tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        node.diameter = compute_diameter(node)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

def distbox(nodeA: ClusterNode, nodeB: ClusterNode) -> float:
    """
    Compute the distance between two axis-aligned bounding boxes
    as in the TP: distance between the closest faces.

    If the boxes overlap along an axis, the contribution along that axis is 0.
    """
    A_min, A_max = nodeA.bbox[:, 0], nodeA.bbox[:, 1]
    B_min, B_max = nodeB.bbox[:, 0], nodeB.bbox[:, 1]

    # x-direction
    dx = max(
        0.0,
        max(A_min[0] - B_max[0],  # box A is to the right of B
            B_min[0] - A_max[0])  # box B is to the right of A
    )

    # y-direction
    dy = max(
        0.0,
        max(A_min[1] - B_max[1],  # box A is above B
            B_min[1] - A_max[1])  # box B is above A
    )

    return float(np.sqrt(dx**2 + dy**2))


def build_balanced_cluster_tree(
    points: np.ndarray,
    Nleaf: int,
    indices: Optional[np.ndarray] = None,
    level: int = 0,
) -> ClusterNode:
    """
    Build a *balanced* binary cluster tree by recursively bisecting along
    the longest box direction at the **median** of the coordinates in that
    direction (so children have similar numbers of points).

    This answers Question 3 of the TP: longest axis + median split.
    """
    if indices is None:
        indices = np.arange(points.shape[0], dtype=int)

    # Build node + bbox
    bbox = compute_bbox(points, indices)
    node = ClusterNode(indices=indices, bbox=bbox, level=level)

    # Stopping criterion: small cluster becomes a leaf
    if indices.size <= Nleaf:
        return node

    # Longest axis of the current bounding box
    extent = node.extent           # bbox[:,1] - bbox[:,0]
    axis = int(np.argmax(extent))

    if extent[axis] <= 0.0:
        # Degenerate box: don't try to split further
        return node

    # Coordinates along this axis
    coords = points[indices, axis]

    # Median-based split (balanced in #points)
    median_val = np.median(coords)
    left_mask = coords <= median_val
    right_mask = ~left_mask

    # Robust fallback: if median doesn't actually split, cut by sorted half
    if np.all(left_mask) or np.all(right_mask):
        order = np.argsort(coords)
        if order.size < 2:
            return node  # nothing reasonable to do
        mid = order.size // 2
        left_idx = indices[order[:mid]]
        right_idx = indices[order[mid:]]
    else:
        left_idx = indices[left_mask]
        right_idx = indices[right_mask]

    # Create children recursively
    node.left = build_balanced_cluster_tree(
        points, Nleaf, indices=left_idx, level=level + 1
    )
    node.right = build_balanced_cluster_tree(
        points, Nleaf, indices=right_idx, level=level + 1
    )

    return node



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Mesh_generator import build_mesh_from_points

    R = 1.0
    n_points = 32
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    mesh = build_mesh_from_points(pts, closed=True)

    #root = build_geometric_cluster_tree(mesh.points, Nleaf=3)
    root = build_balanced_cluster_tree(mesh.points, Nleaf=3)
    precompute_diameters(root)

    print("Total tree depth:", tree_depth(root))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # geometry + boxes
    axes[0].plot(mesh.points[:, 0], mesh.points[:, 1], "o", ms=3)
    plot_boxes(root, axes[0], linewidth=0.8)
    axes[0].set_aspect("equal")
    axes[0].set_title("Geometric partition")

    # actual tree
    plot_cluster_tree(root, ax=axes[1])
    plt.tight_layout()
    plt.show()

