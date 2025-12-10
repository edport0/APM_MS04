# adm_criterion.py
import numpy as np
import matplotlib.pyplot as plt

from Mesh_generator import build_mesh_from_points
from Hmat_clustering import (
    build_geometric_cluster_tree,
    build_balanced_cluster_tree,
    compute_diameter,
    distbox,
    collect_leaves,
)
from solver_BEM import M
from tp_low_rank import aca_complete_pivoting

def eta_admissible(diamX: float, diamY: float, distance: float, eta: float) -> bool:
    """
    η-admissibility criterion from the TP:

        min(diam(X), diam(Y)) < η * dist(X, Y)
    """
    return min(diamX, diamY) < eta * distance

def compression_rate_block(m: int, n: int, r: int) -> float:
    """
    Compression rate for one block:
        dense storage  ~ m * n
        low-rank store ~ r * (m + n)

    rate = (m*n) / (r*(m+n)).
    """
    if r == 0:
        return 0.0
    return (r * (m + n)) / (m * n)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Geometry: choose a simple test boundary (segment or circle)
    # -------------------------------------------------------------------------
    R = 1.0
    n_points = 1025

    # (a) Segment [-R, R] on the x-axis
    x = np.linspace(-R, R, n_points, endpoint=True)
    pts = np.column_stack([x, np.zeros_like(x)])

    # # (b) Uncomment for a circle
    # theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    # pts = np.column_stack([R * np.cos(theta), R * np.sin(theta)])

    mesh = build_mesh_from_points(pts, closed=False)
    midpoints = 0.5 * (mesh.extremities[:, 0, :] + mesh.extremities[:, 1, :])

    # -------------------------------------------------------------------------
    # 2. Build cluster tree (geometric or balanced)
    # -------------------------------------------------------------------------
    Nleaf = 64

    # Geometric midpoint split
    # root = build_geometric_cluster_tree(mesh.points, Nleaf=Nleaf)

    # Balanced (median) split
    root = build_balanced_cluster_tree(midpoints, Nleaf=Nleaf)

    # -------------------------------------------------------------------------
    # 3. Collect leaf boxes
    # -------------------------------------------------------------------------
    leaves = collect_leaves(root)
    n_leaf = len(leaves)
    print(f"Number of leaf boxes: {n_leaf}")

    # Precompute diameters of leaves (used many times)
    leaf_diam = np.array([compute_diameter(node) for node in leaves])

        # -------------------------------------------------------------------------
    # 4. Compute η-admissibility matrix for all leaf pairs
    #    adm_matrix[i, j] = True  -> admissible
    #    non_adm[i, j]    = True  -> *non*-admissible (red square)
    # -------------------------------------------------------------------------
    eta = 3.0
    n_leaf = len(leaves)

    adm_matrix = np.zeros((n_leaf, n_leaf), dtype=bool)
    for i in range(n_leaf):
        for j in range(n_leaf):
            diamX = leaf_diam[i]
            diamY = leaf_diam[j]
            dist_ij = distbox(leaves[i], leaves[j])
            adm_matrix[i, j] = eta_admissible(diamX, diamY, dist_ij, eta)

    from matplotlib.colors import ListedColormap

    # adm_matrix: bool, shape (n_leaf, n_leaf)
    # 1 = admissible (green), 0 = non-admissible (red)
    state = adm_matrix.astype(int)

    cmap = ListedColormap(["red", "green"])  # 0 -> red, 1 -> green

    plt.figure(figsize=(5, 4))
    plt.imshow(state, cmap=cmap, origin="upper", vmin=0, vmax=1)

    plt.title(f"η-admissible blocks (η = {eta})")
    plt.xlabel("j (Y leaf index)")
    plt.ylabel("i (X leaf index)")
    plt.gca().set_aspect("equal")

    # Optional, labeled colorbar
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["non-admissible", "admissible"])

    plt.tight_layout()
    plt.show()


    # -------------------------------------------------------------------------
    # 6. Print a few examples for interpretation
    # -------------------------------------------------------------------------
    for i in range(min(3, n_leaf)):
        for j in range(i + 1, min(i + 4, n_leaf)):
            dX = leaf_diam[i]
            dY = leaf_diam[j]
            dist_ij = distbox(leaves[i], leaves[j])
            adm = eta_admissible(dX, dY, dist_ij, eta)
            print(
                f"({i},{j}) diamX={dX:.3f}, diamY={dY:.3f}, "
                f"dist={dist_ij:.3f} -> admissible={adm}"
            )

        # --- 7. Q4: check low-rank structure of BEM blocks vs η-admissibility ---

    print("\n=== Q4: SVD numerical rank of BEM blocks vs η-admissibility ===")

    # Assemble BEM matrix on the same boundary
    k = np.pi     # you can change this if you want
    n_quad = 2    # Gauss order used in solver_BEM
    n_elem = mesh.extremities.shape[0]

    print(f"Assembling BEM matrix M of size {n_elem} x {n_elem}...")
    A = M(mesh, n_quad, n_elem, k)

    n_leaf = len(leaves)

    # Build list of (leaf_i, leaf_j) pairs for admissible and non-admissible blocks
    adm_pairs = []
    non_adm_pairs = []

    for i in range(n_leaf):
        for j in range(n_leaf):
            if i == j:
                continue  # skip diagonal (always non-low-rank)
            if adm_matrix[i, j]:
                adm_pairs.append((i, j))
            else:
                non_adm_pairs.append((i, j))

    print(f"Total admissible leaf pairs   : {len(adm_pairs)}")
    print(f"Total non-admissible leaf pairs: {len(non_adm_pairs)}")

    # How many blocks we want to test in each category
    n_test_adm = min(5, len(adm_pairs))
    n_test_non = min(5, len(non_adm_pairs))

    # Helper: compute numerical rank from singular values with a given sv tolerance
    def numerical_rank_from_singular_values(S, sv_tol=1e-3):
        if S[0] == 0:
            return 0
        rel = S / S[0]
        return int(np.sum(rel > sv_tol))

    # Helper: analyse one block (i,j) with SVD
    def analyse_block(tag, i_leaf, j_leaf):
        I = leaves[i_leaf].indices
        J = leaves[j_leaf].indices
        A_block = A[np.ix_(I, J)]
        m, n = A_block.shape

        print(f"\n[{tag}] leaf pair ({i_leaf}, {j_leaf}), block size {m} x {n}")

        # SVD
        U, S, Vh = np.linalg.svd(A_block, full_matrices=False)

        # Show first few singular values
        rel_sv = S / S[0]
        n_show = min(10, len(S))
        print("  first singular values σ_k / σ_0:")
        print("   ", "  ".join(f"{rel_sv[k]:.2e}" for k in range(n_show)))

        # Numerical rank for several thresholds on singular values
        for sv_tol in [1e-1, 1e-2, 1e-3]:
            r = numerical_rank_from_singular_values(S, sv_tol=sv_tol)
            print(f"  sv_tol = {sv_tol:4.0e} -> numerical rank r = {r} "
                  f"(r / min(m,n) ≈ {r / min(m, n):.2f})")

    # Randomly pick some admissible blocks and analyse them
    if n_test_adm > 0:
        print(f"\n--- Analysing {n_test_adm} admissible (far) BEM blocks ---")
        # to have reproducible examples
        rng = np.random.default_rng(0)
        chosen_adm = rng.choice(len(adm_pairs), size=n_test_adm, replace=False)
        for idx in chosen_adm:
            i_leaf, j_leaf = adm_pairs[idx]
            analyse_block("admissible / far", i_leaf, j_leaf)
    else:
        print("No admissible off-diagonal leaf pairs found (check η or tree).")

    # Randomly pick some non-admissible blocks and analyse them
    if n_test_non > 0:
        print(f"\n--- Analysing {n_test_non} non-admissible (near) BEM blocks ---")
        rng = np.random.default_rng(1)
        chosen_non = rng.choice(len(non_adm_pairs), size=n_test_non, replace=False)
        for idx in chosen_non:
            i_leaf, j_leaf = non_adm_pairs[idx]
            analyse_block("non-admissible / near", i_leaf, j_leaf)
    else:
        print("No non-admissible off-diagonal leaf pairs found (check η or tree).")

    print("\nInterpretation hint for your report:")
    print("  • η-admissible (far) blocks should show fast singular value decay")
    print("    and small numerical ranks r compared to their size.")
    print("  • non-admissible (near) blocks should need much larger ranks,")
    print("    often comparable to min(block dimensions).")

        # --- 8. Q5: ACA compression on admissible BEM blocks ---

    print("\n=== Q5: ACA compression on η-admissible BEM blocks ===")

    n_leaf = len(leaves)

    # Collect all admissible off-diagonal leaf pairs
    adm_pairs = []
    for i in range(n_leaf):
        for j in range(n_leaf):
            if i == j:
                continue
            if adm_matrix[i, j]:
                adm_pairs.append((i, j))

    print(f"Number of admissible leaf pairs: {len(adm_pairs)}")

    if not adm_pairs:
        print("No admissible pairs found (check η or the tree construction).")
    else:
        # ACA parameters
        eps_aca = 1e-3   # pivot tolerance (you can play with this)
        rmax_aca = None  # or an int if you want to cap the rank

        total_dense = 0.0
        total_lowrank = 0.0

        block_ranks = []
        block_rates = []
        block_errors = []

        # If runtime is too high, you can restrict:
        # adm_pairs = adm_pairs[:50]

        for (i_leaf, j_leaf) in adm_pairs:
            I = leaves[i_leaf].indices
            J = leaves[j_leaf].indices
            A_block = A[np.ix_(I, J)]
            m, n = A_block.shape

            # Run ACA on this block
            U_aca, V_aca, r_aca, rel_res = aca_complete_pivoting(
            A_block, epsilon=eps_aca, max_rank=rmax_aca
            )

            # Storage counts
            dense_storage = m * n
            lowrank_storage = r_aca * (m + n) if r_aca > 0 else 0

            total_dense += dense_storage
            total_lowrank += lowrank_storage

            # Per-block stats
            block_ranks.append(r_aca)
            rate = compression_rate_block(m, n, r_aca)
            block_rates.append(rate)
            block_errors.append(rel_res)

        # Global compression rate over all admissible blocks
        if total_lowrank > 0:
            global_rate = total_lowrank / total_dense
        else:
            global_rate = 0.0

        block_ranks = np.array(block_ranks)
        block_rates = np.array(block_rates)
        block_errors = np.array(block_errors)

        print(f"\nGlobal ACA compression rate on admissible blocks: {global_rate:.2f}")
        print(f"Average ACA rank on admissible blocks           : {block_ranks.mean():.1f}")
        print(f"Average per-block compression rate              : {block_rates.mean():.2f}")
        print(f"Median relative ACA error on admissible blocks  : {np.median(block_errors):.2e}")
        print(f"Max   relative ACA error on admissible blocks   : {block_errors.max():.2e}")



