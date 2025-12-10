import numpy as np
from dataclasses import dataclass
from tree_generator import ClusterNode, build_tree, compute_diameter, distbox
from build_balanced_tree import build_balanced_tree
from adm_criterion import compute_adm


# ---------------------------------------------------------------------
# Block node definition
# ---------------------------------------------------------------------
@dataclass
class BlockNode:
    nodeX: ClusterNode
    nodeY: ClusterNode
    is_admissible: bool = False
    children: list = None
    level: int = 0

    @property
    def is_leaf(self):
        return self.children is None or len(self.children) == 0


# ---------------------------------------------------------------------
# Recursive construction of block-cluster tree
# ---------------------------------------------------------------------
def build_block_tree(nodeX: ClusterNode, nodeY: ClusterNode, eta: float = 3.0, level: int = 0) -> BlockNode:
    diamX = compute_diameter(nodeX)
    diamY = compute_diameter(nodeY)
    distXY = distbox(nodeX, nodeY)
    admissible = compute_adm(diamX, diamY, distXY, eta)

    block = BlockNode(nodeX=nodeX, nodeY=nodeY, is_admissible=admissible, children=[], level=level)

    if admissible or (nodeX.is_leaf and nodeY.is_leaf):
        # Stop here: either admissible or both leaves
        return block

    # Otherwise, need to subdivide
    if nodeX.is_leaf and not nodeY.is_leaf:
        # Subdivide only Y
        for cY in nodeY.children:
            block.children.append(build_block_tree(nodeX, cY, eta, level + 1))
    elif not nodeX.is_leaf and nodeY.is_leaf:
        # Subdivide only X
        for cX in nodeX.children:
            block.children.append(build_block_tree(cX, nodeY, eta, level + 1))
    else:
        # Subdivide both
        for cX in nodeX.children:
            for cY in nodeY.children:
                block.children.append(build_block_tree(cX, cY, eta, level + 1))

    return block


# ---------------------------------------------------------------------
# Utilities for analysis
# ---------------------------------------------------------------------
def collect_blocks(block: BlockNode):
    """Return list of all (admissible/full) leaf blocks."""
    if block.is_leaf:
        return [block]
    blocks = []
    for c in block.children:
        blocks.extend(collect_blocks(c))
    return blocks


def count_admissible(blocks):
    return sum(1 for b in blocks if b.is_admissible)

def centroid(node):
    return 0.5*(node.bbox_min + node.bbox_max)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Mesh_generator import build_mesh_from_points
    import matplotlib.pyplot as plt

    # --- Geometry and tree
    R = 1.0
    n_points = 1000
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    #pts = np.column_stack([R*np.cos(theta), R*np.sin(theta)])
    pts = np.column_stack([np.linspace(-R, R, n_points, endpoint = True), np.zeros(n_points)])
    mesh = build_mesh_from_points(pts)
    Nleaf = 64
    #root = build_tree(mesh.points, Nleaf)
    root = build_balanced_tree(mesh.points, Nleaf)

    # --- Build block-cluster tree
    eta = 3.0
    block_root = build_block_tree(root, root, eta)

    # --- Analyze and display statistics
    all_blocks = collect_blocks(block_root)
    n_adm = count_admissible(all_blocks)
    print(f"Total blocks: {len(all_blocks)} | admissible: {n_adm} | full: {len(all_blocks)-n_adm}")

        # --------------------------------------------------------------
    # 5. Visualization: full hierarchical block coverage (no gaps)
    # --------------------------------------------------------------
    import matplotlib.patches as patches

    # --- collect all leaves
    leaves = []
    def collect_leaves(node):
        if node.is_leaf:
            leaves.append(node)
        else:
            for c in node.children:
                collect_leaves(c)
    collect_leaves(root)

    # sort leaves geometrically (optional but helps readability)
    def centroid(node): return 0.5 * (node.bbox_min + node.bbox_max)
    leaves = sorted(leaves, key=lambda n: centroid(n)[0])  # sort by x

    # --- leaf index mapping
    leaf_index = {id(n): k for k, n in enumerate(leaves)}

    # --- compute index spans for *all* nodes (leaves + internal)
    node_span = {}

    def compute_node_span(node):
        if node.is_leaf:
            idx = leaf_index[id(node)]
            node_span[id(node)] = (idx, idx + 1)
        else:
            for c in node.children:
                compute_node_span(c)
            l0, l1 = node_span[id(node.children[0])]
            r0, r1 = node_span[id(node.children[1])]
            node_span[id(node)] = (l0, r1)

    compute_node_span(root)

    # --- build figure
    nL = len(leaves)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, nL)
    ax.set_ylim(0, nL)
    ax.set_aspect('equal')
    ax.set_xlabel("X block index")
    ax.set_ylabel("Y block index")
    ax.set_title(f"Block-cluster structure (η={eta})")

    # --- draw every block, colored by admissibility
    for b in all_blocks:
        i0, i1 = node_span[id(b.nodeX)]
        j0, j1 = node_span[id(b.nodeY)]
        color = 'green' if b.is_admissible else 'red'
        rect = patches.Rectangle(
            (i0, j0), i1 - i0, j1 - j0,
            facecolor=color, edgecolor='black', lw=0.5
        )
        ax.add_patch(rect)

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
