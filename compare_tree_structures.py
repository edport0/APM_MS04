import numpy as np
import matplotlib.pyplot as plt
from Mesh_generator import build_mesh_from_points
from tree_generator import build_tree as build_geometric_tree
from build_balanced_tree import build_balanced_tree


# ---------------------------------------------------------------------
# Compact tree plotting utilities
# ---------------------------------------------------------------------
def extract_coords(root, depth=0, x_offset=0.0, width=1.0, nodes=None, edges=None):
    """Recursively assign (x, y) coordinates to nodes."""
    if nodes is None: nodes, edges = [], []
    nodes.append((root, x_offset + width/2, -depth))
    if len(root.children) == 2:
        left, right = root.children
        edges.append(((x_offset + width/2, -depth), (x_offset + width/4, -(depth+1))))
        edges.append(((x_offset + width/2, -depth), (x_offset + 3*width/4, -(depth+1))))
        extract_coords(left, depth+1, x_offset, width/2, nodes, edges)
        extract_coords(right, depth+1, x_offset+width/2, width/2, nodes, edges)
    return nodes, edges


def fast_plot_tree(ax, root, title):
    nodes, edges = extract_coords(root)
    ex = np.array([[e[0][0], e[1][0]] for e in edges])
    ey = np.array([[e[0][1], e[1][1]] for e in edges])
    ax.plot(ex.T, ey.T, 'k-', lw=0.7, alpha=0.7)
    ax.set_title(title)
    ax.axis('off')


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Generate geometry
    R = 1.0
    n_points = 1000
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    #pts = np.column_stack([R*np.cos(theta), R*np.sin(theta)])
    pts = np.column_stack([np.linspace(-R, R, n_points, endpoint = True), np.zeros(n_points)])
    mesh = build_mesh_from_points(pts)

    Nleaf = 8
    geom_root = build_geometric_tree(mesh.points, Nleaf)
    bal_root = build_balanced_tree(mesh.points, Nleaf)

    # Side-by-side tree visualisation
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fast_plot_tree(axs[0], bal_root, "Balanced (median split)")
    fast_plot_tree(axs[1], geom_root, "Geometric split")
    plt.tight_layout()
    plt.show()
