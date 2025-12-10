import time
import numpy as np

from Mesh_generator import build_mesh_from_points
from Hmat_clustering import (
    build_balanced_cluster_tree,
    collect_leaves,
)
from solver_BEM import M, b
from Hmat_final import (
    build_Hmatrix_from_BEM_partialACA,
    Hmatvec,
)


def compute_Hmatrix_compression(H, leaves):
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
    h_storage_total = 0.0

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
                h_storage_total += r * (m + n)
                ranks.append(r)
                num_lr += 1
            elif block.is_dense:
                h_storage_total += m * n
                num_dense += 1
            else:
                # empty block: no storage (should not normally happen)
                continue

    compression_rate = dense_total / h_storage_total if h_storage_total > 0 else 0.0
    avg_rank = float(np.mean(ranks)) if ranks else 0.0

    return compression_rate, avg_rank, num_lr, num_dense


# Simple GMRES implementation (no restart, full storage)

def gmres(Aop, b, x0=None, tol=1e-8, maxiter=None, verbose=False):
    """

    Parameters
    ----------
    Aop : callable
        Linear operator: given x (shape (N,)), returns A @ x (shape (N,)).
    b : (N,) ndarray
        Right-hand side.
    x0 : (N,) ndarray or None
        Initial guess (default = 0).
    tol : float
        Relative residual tolerance: stop when ||r|| / ||b|| <= tol.
    maxiter : int or None
        Maximum number of iterations (default = N).
    verbose : bool
        If True, print residual each iteration.

    Returns
    -------
    x : (N,) ndarray
        Approximate solution.
    it : int
        Number of iterations used.
    res_hist : list of float
        Residual history: ||r_k|| / ||b|| at each iteration.
    """
    b = np.asarray(b, dtype=np.complex128)
    N = b.size

    if maxiter is None:
        maxiter = N

    if x0 is None:
        x0 = np.zeros(N, dtype=np.complex128)
    else:
        x0 = np.asarray(x0, dtype=np.complex128)

    # Initial residual
    r0 = b - Aop(x0)
    beta = np.linalg.norm(r0)
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return x0, 0, [0.0]

    rel0 = beta / b_norm
    res_hist = [rel0]
    if verbose:
        print(f"GMRES iter 0: rel_res = {rel0:.3e}")

    if rel0 <= tol:
        return x0, 0, res_hist

    # Arnoldi basis and Hessenberg matrix
    V = np.zeros((N, maxiter + 1), dtype=np.complex128)
    H = np.zeros((maxiter + 1, maxiter), dtype=np.complex128)

    V[:, 0] = r0 / beta

    for j in range(maxiter):
        # w = A v_j
        w = Aop(V[:, j])

        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)   # inner product
            w -= H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] != 0 and j + 1 < maxiter:
            V[:, j + 1] = w / H[j + 1, j]

        # Solve least squares min ||beta e1 - H_{0:j+1,0:j+1} y||
        e1 = np.zeros(j + 2, dtype=np.complex128)
        e1[0] = beta
        Hj = H[0:j + 2, 0:j + 1]
        y, *_ = np.linalg.lstsq(Hj, e1, rcond=None)

        # Form approximate solution
        x = x0 + V[:, :j + 1] @ y
        r = b - Aop(x)
        rel_res = np.linalg.norm(r) / b_norm
        res_hist.append(rel_res)

        if verbose:
            print(f"GMRES iter {j+1}: rel_res = {rel_res:.3e}")

        if rel_res <= tol:
            return x, j + 1, res_hist

    # Did not converge within maxiter
    return x, maxiter, res_hist


if __name__ == "__main__":
    
    R = 1.0
    n_points = 1025          # number of boundary points (nodes)
    x_coord = np.linspace(-R, R, n_points, endpoint=True)
    pts = np.column_stack([x_coord, np.zeros_like(x_coord)])

    # Open segment [-R, R] on the x-axis
    mesh = build_mesh_from_points(pts, closed=False)
    n_elem = mesh.extremities.shape[0]
    print(f"Number of boundary elements (DOFs): {n_elem}")

    # Midpoints used for clustering (one per element / DOF)
    midpoints = 0.5 * (mesh.extremities[:, 0, :] + mesh.extremities[:, 1, :])

    Nleaf = 64
      # leaf size; feel free to experiment (64, 128, 256, ...)
    root = build_balanced_cluster_tree(midpoints, Nleaf=Nleaf)
    leaves = collect_leaves(root)
    print(f"Number of leaf boxes in cluster tree: {len(leaves)}")

    k = np.pi   # wavenumber
    n_quad = 2  # Gauss order

    print(f"Assembling dense BEM matrix A of size {n_elem} x {n_elem} ...")
    t0 = time.perf_counter()
    A = M(mesh, n_quad, n_elem, k)
    t1 = time.perf_counter()
    t_d = t1 - t0
    print(f"Done in {t_d:.3f} s.")

    print("Assembling RHS b ...")
    B = -b(k, mesh, n_quad)   # sign consistent with solver_BEM main
    print("Done.\n")

    eta = 3.0
    eps_aca = 1e-3

    print("Building H-matrix with partial ACA (no full A used inside) ...")
    t0 = time.perf_counter()
    H = build_Hmatrix_from_BEM_partialACA(
        mesh,
        k,
        n_quad,
        leaves,
        eta,
        eps_aca=eps_aca,
        rmax_aca=5,
    )
    t1 = time.perf_counter()
    t_aca = t1 - t0
    print(f"H-matrix assembly done in {t_aca:.3f} s.\n")

    comp_rate, avg_rank, num_lr, num_dense = compute_Hmatrix_compression(H, leaves)
    print("H-matrix compression stats:")
    print(f"  low-rank blocks : {num_lr}")
    print(f"  dense blocks    : {num_dense}")
    print(f"  average rank    : {avg_rank:.2f}")
    print(f"  compression     : {comp_rate:.2f}x (dense_storage / H_storage)\n")

    print("=== GMRES with dense matrix A ===")
    def A_dense_mv(x):
        return A @ x

    tol = 1e-8
    maxiter = 200

    t0 = time.perf_counter()
    x_dense, it_dense, res_dense_hist = gmres(
        A_dense_mv, B, tol=tol, maxiter=maxiter, verbose=False
    )
    t1 = time.perf_counter()
    dense_time = t1 - t0

    print(f"Dense GMRES: iterations = {it_dense}, "
          f"final rel_res = {res_dense_hist[-1]:.3e}, "
          f"time = {dense_time:.3f} s\n")

    print("=== GMRES with H-matrix (Hmatvec) ===")
    def A_h_mv(x):
        return Hmatvec(H, x)

    t0 = time.perf_counter()
    x_h, it_h, res_h_hist = gmres(
        A_h_mv, B, tol=tol, maxiter=maxiter, verbose=False
    )
    t1 = time.perf_counter()
    h_time = t1 - t0

    print(f"H-matrix GMRES: iterations = {it_h}, "
          f"final rel_res = {res_h_hist[-1]:.3e}, "
          f"time = {h_time:.3f} s\n")

    rel_diff = np.linalg.norm(x_h - x_dense) / np.linalg.norm(x_dense)
    print(f"Relative difference between solutions ||x_H - x_dense|| / ||x_dense|| = {rel_diff:.3e}")

    print("\nSummary:")
    print(f"  dense  : it = {it_dense:3d}, time = {dense_time:.3f} s, "
          f"final rel_res = {res_dense_hist[-1]:.3e}")
    print(f"  H-mat  : it = {it_h:3d}, time = {h_time:.3f} s, "
          f"final rel_res = {res_h_hist[-1]:.3e}")
    print(f"  solution relative difference = {rel_diff:.3e}")

    total_dense = dense_time + t_d
    total_aca = h_time + t_aca

    print(f"Total operation time for the naive system = {total_dense:3d}")
    print(f"Total operation time for ACA compressed system = {total_aca:3d}")