import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time
from Mesh_generator import build_mesh_from_points
from Hmat_clustering import (
    ClusterNode,
    build_balanced_cluster_tree,
    collect_leaves,
    compute_diameter,
    distbox,
    precompute_diameters,
)
from solver_BEM import M, green, quad_segment, calculate_Mii_regular, calculate_Mii_singular
from tp_low_rank import aca_partial_pivoting, aca_complete_pivoting
from adm_criterion import eta_admissible

@dataclass
class HBlock:
    """One leaf-×-leaf block in the H-matrix."""
    dense: Optional[np.ndarray] = None  # non-admissible block stored dense
    U: Optional[np.ndarray] = None      # admissible block: U
    V: Optional[np.ndarray] = None      # admissible block: V

    @property
    def is_lowrank(self) -> bool:
        return (self.U is not None) and (self.V is not None)

    @property
    def is_dense(self) -> bool:
        return self.dense is not None

@dataclass
class HMatrixBlock:
    """
    One block in the H-matrix block tree.

    Attributes
    ----------
    I, J : 1D integer arrays
        Global row and column index sets for this block.
    dense : (|I|, |J|) ndarray or None
        Dense representation of the block (non-admissible leaf).
    U, V : ndarrays or None
        Low-rank factors for admissible leaf: A_block ≈ U @ V.T
        with U shape (|I|, r), V shape (|J|, r).
    children : list of HMatrixBlock
        4 children for internal nodes (2x2 subdivision), empty for leaves.
    """
    I: np.ndarray
    J: np.ndarray
    dense: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    children: List["HMatrixBlock"] = field(default_factory=list)

    @property
    def is_lowrank(self) -> bool:
        return (self.U is not None) and (self.V is not None)

    @property
    def is_dense(self) -> bool:
        return self.dense is not None


@dataclass
class HMatrix:
    """
    Simple leaf-level H-matrix structure:
      - leaves: list of cluster nodes (with .indices),
      - blocks[i][j]: HBlock for leaf pair (i,j),
      - N: total number of DOFs.
    """
    leaves: List
    blocks: List[List[HBlock]]
    N: int

def bem_entry(mesh, k: float, n_quad: int, i: int, j: int) -> complex:
    """
    Return a single BEM matrix entry A_{i,j} for given element indices i, j.

    This duplicates the logic of solver_BEM.M but only for one pair (i,j).
    """
    Y0i = mesh.extremities[i, 0, :]
    Y1i = mesh.extremities[i, 1, :]
    Y0j = mesh.extremities[j, 0, :]
    Y1j = mesh.extremities[j, 1, :]

    if i != j:
        # Off-diagonal: standard double integral ∫_{Γ_i} ∫_{Γ_j} G
        def integral_element_q(points_on_i):
            res = np.zeros(points_on_i.shape[0], dtype=np.complex128)
            for idx, zi in enumerate(points_on_i):
                integrand = lambda points_on_j: green(zi, points_on_j, k)
                res[idx] = quad_segment(integrand, Y0j, Y1j, n_quad)
            return res

        return quad_segment(integral_element_q, Y0i, Y1i, n_quad)

    else:
        # Diagonal: regular + singular parts
        A_reg = calculate_Mii_regular(Y0i, Y1i, k, n_quad)
        A_sing = calculate_Mii_singular(Y0i, Y1i, n_quad)
        return A_reg + A_sing
    
def aca_partial_BEM_block(mesh,
                          k: float,
                          n_quad: int,
                          I: np.ndarray,
                          J: np.ndarray,
                          epsilon: float = 0.0,
                          max_rank: Optional[int] = None):
    """
    Parameters
    ----------
    mesh, k, n_quad : BEM data.
    I, J : 1D integer arrays
        Global indices of rows and columns in this block.
    epsilon : float
        Pivot tolerance. If 0.0, do not stop on pivot size (only max_rank).
    max_rank : int or None
        Maximum rank. If None, use min(m, n).

    Returns
    -------
    U : (m, r) ndarray
    V : (n, r) ndarray
    r : int
        Final rank.
    """
    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)

    m = I.size
    n = J.size

    if m == 0 or n == 0:
        return (np.zeros((m, 0), dtype=np.complex128),
                np.zeros((n, 0), dtype=np.complex128),
                0)
    if max_rank is None:
        max_rank = min(m, n)

    U = None
    V = None
    r = 0

    # convenience: build full row/column of A_{IJ}
    def build_row(i_loc: int) -> np.ndarray:
        """Row i_loc of A_{IJ} as a vector of length n."""
        gi = I[i_loc]
        row = np.empty(n, dtype=np.complex128)
        for jj in range(n):
            row[jj] = bem_entry(mesh, k, n_quad, gi, J[jj])
        return row

    def build_col(j_loc: int) -> np.ndarray:
        """Column j_loc of A_{IJ} as a vector of length m."""
        gj = J[j_loc]
        col = np.empty(m, dtype=np.complex128)
        for ii in range(m):
            col[ii] = bem_entry(mesh, k, n_quad, I[ii], gj)
        return col

    # initial row index: take simply i = 0 (could be improved)
    i_loc = 0

    while r < max_rank:
        # residual row r = A[i_loc, :] - sum_{ell} U[i_loc, ell] V[:, ell]^T
        row = build_row(i_loc)
        if U is not None:
            row = row - U[i_loc, :] @ V.T   # (r,) @ (r,n) -> (n,)

        # pick column j with max |row|
        j_loc = int(np.argmax(np.abs(row)))

        # residual column c = A[:, j_loc] - sum_{ell} U[:, ell] V[j_loc, ell]^*
        col = build_col(j_loc)
        if U is not None:
            col = col - U @ V[j_loc, :].conj().T  # (m,r) @ (r,) -> (m,)

        pivot = col[i_loc]

        # optional stopping on pivot size (very rough)
        if epsilon > 0.0:
            # scale by row norm (no Frobenius norm needed)
            if np.abs(pivot) <= epsilon * np.linalg.norm(row):
                break

        # rank-one factors
        u_k = col
        v_k = row / pivot

        if U is None:
            U = u_k[:, None]
            V = v_k[:, None]
        else:
            U = np.column_stack((U, u_k))
            V = np.column_stack((V, v_k))

        r += 1

        # next row index from |u_k|
        i_loc = int(np.argmax(np.abs(u_k)))

    if U is None:
        U = np.zeros((m, 0), dtype=np.complex128)
        V = np.zeros((n, 0), dtype=np.complex128)
        r = 0

    return U, V, r


def assemble_BEM_block(mesh,
                       k: float,
                       n_quad: int,
                       I: np.ndarray,
                       J: np.ndarray) -> np.ndarray:
    """""

    Parameters
    ----------
    mesh : boundary mesh (from Mesh_generator)
    k : float
        Wavenumber.
    n_quad : int
        Quadrature order per segment.
    I, J : 1D integer arrays
        Global element indices (row and column clusters).

    Returns
    -------
    A_block : (len(I), len(J)) ndarray (complex)
        Submatrix [A_{p,q}]_{p in I, q in J}.
    """
    I = np.asarray(I, dtype=int)
    J = np.asarray(J, dtype=int)

    m = I.size
    n = J.size
    A_block = np.zeros((m, n), dtype=np.complex128)

    for ii, p in enumerate(I):
        Y0p = mesh.extremities[p, 0, :]
        Y1p = mesh.extremities[p, 1, :]

        for jj, q in enumerate(J):
            Y0q = mesh.extremities[q, 0, :]
            Y1q = mesh.extremities[q, 1, :]

            if p != q:
                # Off-diagonal: standard double integral over Γ_p × Γ_q
                def integral_element_q(points_on_p):
                    res = np.zeros(points_on_p.shape[0], dtype=np.complex128)
                    for idx, zp in enumerate(points_on_p):
                        integrand = lambda points_on_q: green(zp, points_on_q, k)
                        res[idx] = quad_segment(integrand, Y0q, Y1q, n_quad)
                    return res

                A_block[ii, jj] = quad_segment(integral_element_q, Y0p, Y1p, n_quad)

            else:
                # Diagonal: same semi-analytic treatment as in solver_BEM.M
                A_reg = calculate_Mii_regular(Y0p, Y1p, k, n_quad)
                A_sing = calculate_Mii_singular(Y0p, Y1p, n_quad)
                A_block[ii, jj] = A_reg + A_sing

    return A_block

def build_Hmatrix_from_dense(A: np.ndarray,
                             leaves,
                             eta: float,
                             eps_aca: float = 1e-3,
                             rmax_aca: Optional[int] = None) -> HMatrix:
    """
    Assemble a leaf-level H-matrix A_h from the dense matrix A.

    For each leaf pair (i, j):
      - compute η-admissibility from leaf diameters and bbox distances,
      - if admissible (and off-diagonal): approximate A_{IJ} by ACA low rank U V^T,
      - else: store A_{IJ} as a dense block.
    """
    N = A.shape[0]
    n_leaf = len(leaves)

    # Precompute diameters for all leaves (using Hmat_clustering.compute_diameter)
    leaf_diam = np.array([compute_diameter(node) for node in leaves])

    # Build η-admissibility matrix using your criterion from adm_criterion
    adm_matrix = np.zeros((n_leaf, n_leaf), dtype=bool)
    for i in range(n_leaf):
        for j in range(n_leaf):
            diamX = leaf_diam[i]
            diamY = leaf_diam[j]
            dist_ij = distbox(leaves[i], leaves[j])
            adm_matrix[i, j] = eta_admissible(diamX, diamY, dist_ij, eta)

    # Allocate blocks
    blocks: List[List[HBlock]] = [[HBlock() for _ in range(n_leaf)]
                                  for _ in range(n_leaf)]

    # Fill blocks: use ACA for admissible off-diagonal, dense otherwise
    for i in range(n_leaf):
        I = leaves[i].indices
        for j in range(n_leaf):
            J = leaves[j].indices
            A_block = A[np.ix_(I, J)]
            m, n = A_block.shape

            if adm_matrix[i, j] and (i != j):
                # Admissible off-diagonal: try low-rank ACA
                U_aca, V_aca, r_aca, rel_res = aca_complete_pivoting(
                    A_block, epsilon=eps_aca, max_rank=rmax_aca
                )

                # keep only if it actually compresses
                if r_aca > 0 and r_aca * (m + n) < m * n:
                    blocks[i][j] = HBlock(dense=None, U=U_aca, V=V_aca)
                else:
                    blocks[i][j] = HBlock(dense=A_block.copy())
            else:
                # Non-admissible or diagonal: store dense
                blocks[i][j] = HBlock(dense=A_block.copy())

    return HMatrix(leaves=leaves, blocks=blocks, N=N)

def build_Hmatrix_from_BEM_partialACA(mesh,
                                      k: float,
                                      n_quad: int,
                                      leaves,
                                      eta: float,
                                      eps_aca: float = 1e-3,
                                      rmax_aca: Optional[int] = None) -> HMatrix:
    """"
    Parameters
    ----------
    mesh : boundary mesh
    k : float
        Wavenumber.
    n_quad : int
        Quadrature order per segment.
    leaves : list of ClusterNode
        Leaf clusters built on DOF midpoints.
    eta : float
        η parameter.
    eps_aca : float
        Pivot tolerance in partial ACA (per block).
    rmax_aca : int or None
        Maximum rank allowed in ACA (None => min(m, n)).

    Returns
    -------
    H : HMatrix
        Assembled hierarchical matrix representation.
    """
    N = mesh.extremities.shape[0]
    n_leaf = len(leaves)

    # Precompute diameters and η-admissibility
    leaf_diam = np.array([compute_diameter(node) for node in leaves])
    adm_matrix = np.zeros((n_leaf, n_leaf), dtype=bool)
    for i in range(n_leaf):
        for j in range(n_leaf):
            diamX = leaf_diam[i]
            diamY = leaf_diam[j]
            dist_ij = distbox(leaves[i], leaves[j])
            adm_matrix[i, j] = eta_admissible(diamX, diamY, dist_ij, eta)

    blocks: List[List[HBlock]] = [[HBlock() for _ in range(n_leaf)]
                                  for _ in range(n_leaf)]

    for i in range(n_leaf):
        I = leaves[i].indices
        for j in range(n_leaf):
            J = leaves[j].indices

            if adm_matrix[i, j] and (i != j):
                # η-admissible off-diagonal: LOW-RANK via matrix-free partial ACA
                U_aca, V_aca, r_aca = aca_partial_BEM_block(
                    mesh, k, n_quad, I, J,
                    epsilon=eps_aca,
                    max_rank=rmax_aca,
                )
                if r_aca > 0:
                    blocks[i][j] = HBlock(dense=None, U=U_aca, V=V_aca)
                else:
                    # degenerate case: store dense zero block (or compute it once)
                    # to keep things simple, treat as dense and assemble
                    A_block = assemble_BEM_block(mesh, k, n_quad, I, J)
                    blocks[i][j] = HBlock(dense=A_block)
            else:
                # Non-admissible or diagonal: DENSE block
                A_block = assemble_BEM_block(mesh, k, n_quad, I, J)
                blocks[i][j] = HBlock(dense=A_block)

    return HMatrix(leaves=leaves, blocks=blocks, N=N)


def Hmatvec(H: HMatrix, x: np.ndarray) -> np.ndarray:
    """
    Matrix-vector product y = A_h x, where A_h is stored as an HMatrix.

    Uses:
      - low-rank factors U, V on admissible blocks,
      - dense blocks on non-admissible blocks.
    """
    N = H.N
    y = np.zeros(N, dtype=x.dtype)

    leaves = H.leaves
    blocks = H.blocks
    n_leaf = len(leaves)

    for i in range(n_leaf):
        I = leaves[i].indices
        for j in range(n_leaf):
            J = leaves[j].indices
            block = blocks[i][j]

            if block.is_lowrank:
                # y[I] += U (V^T x[J])
                tmp = block.V.T @ x[J]       # shape (r,)
                y[I] += block.U @ tmp        # (|I|, r) @ (r,)
            elif block.is_dense:
                # y[I] += A_block x[J]
                y[I] += block.dense @ x[J]
            else:
                # empty block (not really used here)
                continue

    return y



def check_Hmatvec_accuracy(H_root: HMatrixBlock,
                           A_dense: np.ndarray,
                           n_tests: int = 5,
                           verbose: bool = True) -> float:
    """
    Parameters
    ----------
    H_root : HMatrixBlock
        Root of the H-matrix representation A_h.
    A_dense : (N, N) ndarray
        Dense reference matrix (e.g. full BEM matrix).
    n_tests : int
        Number of random test vectors x.
    verbose : bool
        If True, print the relative errors for each test.

    Returns
    -------
    max_rel_err : float
        Maximum relative error observed over all tests:
            max_x ||Hmatvec(H_root, x) - A_dense x||_2 / ||A_dense x||_2.
    """
    N = A_dense.shape[0]
    max_rel_err = 0.0

    for t in range(n_tests):
        # Random test vector
        x = np.random.randn(N) + 1j * np.random.randn(N)

        y_ref = A_dense @ x
        y_h   = Hmatvec(H_root, x)

        num = np.linalg.norm(y_h - y_ref)
        den = np.linalg.norm(y_ref)
        rel_err = num / den if den > 0 else 0.0

        max_rel_err = max(max_rel_err, rel_err)

        if verbose:
            print(f"test {t+1}/{n_tests}: relative error = {rel_err:.3e}")

    if verbose:
        print(f"Maximum relative error over {n_tests} tests: {max_rel_err:.3e}")

    return max_rel_err

import time

def benchmark_matvec(A_dense: np.ndarray,
                     H: HMatrix,
                     n_tests: int = 5,
                     n_warmup: int = 2):
    """
    Compare execution time of dense matvec vs H-matrix matvec
    (assembly time NOT included).

    Parameters
    ----------
    A_dense : (N, N) ndarray
        Full dense matrix.
    H : HMatrix
        Assembled H-matrix.
    n_tests, n_warmup : int
        Number of timed and warm-up runs.

    Returns
    -------
    avg_dense, avg_h : float
        Average times in seconds.
    """
    N = A_dense.shape[0]

    # warmup
    for _ in range(n_warmup):
        x = np.random.randn(N) + 1j * np.random.randn(N)
        _ = A_dense @ x
        _ = Hmatvec(H, x)

    dense_times = []
    h_times = []

    for _ in range(n_tests):
        x = np.random.randn(N) + 1j * np.random.randn(N)

        t0 = time.perf_counter()
        y_dense = A_dense @ x
        t1 = time.perf_counter()
        dense_times.append(t1 - t0)

        t0 = time.perf_counter()
        y_h = Hmatvec(H, x)
        t1 = time.perf_counter()
        h_times.append(t1 - t0)

        # optional accuracy check per run
        rel_err = np.linalg.norm(y_h - y_dense) / np.linalg.norm(y_dense)
        print(f"  rel error (run) = {rel_err:.2e}")

    avg_dense = float(np.mean(dense_times))
    avg_h = float(np.mean(h_times))

    print(f"\nAverage dense matvec time   : {avg_dense:.4e} s")
    print(f"Average H-matrix matvec time: {avg_h:.4e} s")
    if avg_h > 0:
        print(f"Speed-up (dense / H)        : {avg_dense / avg_h:.2f}x")

    return avg_dense, avg_h


def compute_Hmatrix_compression(H: HMatrix, leaves):
    """
    Returns
    -------
    compression_rate : float
        (dense storage) / (H storage).
    avg_rank : float
        Average rank over all low-rank blocks.
    num_lr : int
        Number of low-rank blocks.
    num_dense : int
        Number of dense blocks.
    """
    N = H.N
    blocks = H.blocks
    n_leaf = len(leaves)

    dense_total = N * N
    H_storage = 0.0

    ranks = []
    num_lr = 0
    num_dense = 0

    for i in range(n_leaf):
        I = leaves[i].indices
        m = len(I)
        for j in range(n_leaf):
            J = leaves[j].indices
            n = len(J)
            block = blocks[i][j]

            if block.is_lowrank:
                r = block.U.shape[1]
                H_storage += r * (m + n)
                ranks.append(r)
                num_lr += 1
            elif block.is_dense:
                H_storage += m * n
                num_dense += 1
            else:
                # empty block: no storage
                continue

    compression_rate = dense_total / H_storage if H_storage > 0 else 0.0
    avg_rank = float(np.mean(ranks)) if ranks else 0.0

    return compression_rate, avg_rank, num_lr, num_dense



if __name__ == "__main__":
    
    R = 1.0
    n_points = 2048

    x_coord = np.linspace(-R, R, n_points, endpoint=True)
    pts = np.column_stack([x_coord, np.zeros_like(x_coord)])

    mesh = build_mesh_from_points(pts, closed=False)
    n_elems = mesh.extremities.shape[0]
    midpoints = 0.5 * (mesh.extremities[:, 0, :] + mesh.extremities[:, 1, :])

    # Cluster tree on DOFs (midpoints)
    Nleaf = 64
    root = build_balanced_cluster_tree(midpoints, Nleaf=Nleaf)
    leaves = collect_leaves(root)
    n_leaf = len(leaves)
    print(f"Number of leaf boxes: {n_leaf}")

    # Precompute diameters
    leaf_diam = np.array([compute_diameter(node) for node in leaves])

    # η-admissibility matrix
    eta = 3.0
    eps_aca = 1e-3
    rmax_aca = 5

    # BEM matrix A from solver_BEM
    k = np.pi
    n_quad = 2
    n_elem = mesh.extremities.shape[0]

    print("\nAssembling full dense BEM matrix A with M(...) ...")
    t0 = time.perf_counter()
    A = M(mesh, n_quad, n_elems, k)
    t1 = time.perf_counter()
    t_dense_assembly = t1 - t0
    print(f"Dense M assembly time: {t_dense_assembly:.3f} s")

    # 2) Matrix-free partial ACA H-matrix assembly
    print("\nAssembling H-matrix A_h with matrix-free partial ACA ...")
    t0 = time.perf_counter()
    H_partial = build_Hmatrix_from_BEM_partialACA(
        mesh=mesh,
        k=k,
        n_quad=n_quad,
        leaves=leaves,
        eta=eta,
        eps_aca=eps_aca,
        rmax_aca=rmax_aca,
    )
    t1 = time.perf_counter()
    t_H_assembly = t1 - t0
    print(f"H-matrix assembly time: {t_H_assembly:.3f} s")

    comp_rate, avg_rank, num_lr, num_dense = compute_Hmatrix_compression(H_partial, leaves)
    print("\nH-matrix compression statistics:")
    print(f"  low-rank blocks      : {num_lr}")
    print(f"  dense blocks         : {num_dense}")
    print(f"  average low-rank r   : {avg_rank:.2f}")
    print(f"  compression (N^2 / H): {comp_rate:.2f}x")

    # 2) Accuracy check for one vector
    x = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
    y_dense = A @ x
    y_h = Hmatvec(H_partial, x)
    rel_err = np.linalg.norm(y_h - y_dense) / np.linalg.norm(y_dense)
    print(f"Single-test relative error ||H x - A x|| / ||A x|| = {rel_err:.3e}")

    # 3) Question 3 timings
    print("\n=== Timing dense vs H-matrix matvec (Q3) ===")
    benchmark_matvec(A, H_partial, n_tests=5, n_warmup=2)
