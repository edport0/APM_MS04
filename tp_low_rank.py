#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP3: Low-rank factorizations (SVD / ACA) + application to BEM block.

This file is intended as a cleaned-up .py version of the Jupyter notebook
you used for the Low Rank TP. It provides:

1. Generation of synthetic low-rank matrices.
2. Truncated SVD approximation and error analysis.
3. ACA with complete pivoting.
4. ACA with partial pivoting.
5. Simple tests on random low-rank matrices.
6. A test on one off-diagonal block of the BEM matrix from solver_BEM.py.
"""

import numpy as np
import time

# ---------------------------------------------------------------------------
# 1. Generate matrices of (approximately) low rank
# ---------------------------------------------------------------------------

def generate_low_rank_matrix(m, n, r, noise_level=0.0, complex_valued=False, seed=None):
    """
    Generate an m x n matrix of (approximate) rank r:

        A_true = U @ V^T

    and optionally add a small noise term so that the *exact* rank is full,
    but the *numerical* rank is r.

    Parameters
    ----------
    m, n : int
        Dimensions of the matrix.
    r : int
        Target rank (r <= min(m, n)).
    noise_level : float
        Standard deviation of Gaussian noise added to A_true.
    complex_valued : bool
        If True, use complex Gaussian entries.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    A : (m, n) ndarray
        Generated matrix of approximate rank r.
    U_gen, V_gen : ndarrays
        Factors used to generate the rank-r part, U_gen shape (m, r),
        V_gen shape (n, r).
    """
    if seed is not None:
        np.random.seed(seed)

    if complex_valued:
        U_gen = np.random.randn(m, r) + 1j * np.random.randn(m, r)
        V_gen = np.random.randn(n, r) + 1j * np.random.randn(n, r)
    else:
        U_gen = np.random.randn(m, r)
        V_gen = np.random.randn(n, r)

    A_true = U_gen @ V_gen.T

    if noise_level > 0.0:
        if complex_valued:
            noise = noise_level * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
        else:
            noise = noise_level * np.random.randn(m, n)
        A = A_true + noise
    else:
        A = A_true

    return A, U_gen, V_gen


# ---------------------------------------------------------------------------
# 2. Truncated SVD for low-rank approximation
# ---------------------------------------------------------------------------

def truncated_svd(A, tol=None, max_rank=None):
    """
    Low-rank approximation A ≈ U_k V_k^T using SVD.

    We compute
        A = U Σ V^H
    and keep only the first k singular values, where k is chosen either
    from a relative Frobenius tolerance tol, or by max_rank.

    Returns factors (U_k Σ_k, V_k) so that:
        A_k = (U_k Σ_k) @ V_k^T.

    Parameters
    ----------
    A : (m, n) ndarray
        Matrix to compress.
    tol : float or None
        Relative Frobenius error tolerance. If not None, choose the smallest
        k such that ||A - A_k||_F / ||A||_F <= tol.
    max_rank : int or None
        Maximum allowed rank. If tol is None, we keep min(max_rank, min(m, n)).
        If both tol and max_rank are None, we keep the full rank.

    Returns
    -------
    U_fac : (m, k) ndarray
        U_k Σ_k (left factor with singular values absorbed).
    V_fac : (n, k) ndarray
        V_k (right factor).
    rel_error : float
        Achieved relative Frobenius error.
    """
    A = np.array(A, dtype=np.complex128)
    m, n = A.shape

    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    s2 = S**2
    total = np.sum(s2)

    if tol is not None and tol > 0.0:
        # choose smallest k such that sum_{i<=k} s_i^2 >= (1 - tol^2) * total
        cum = np.cumsum(s2)
        target = (1.0 - tol**2) * total
        k = int(np.searchsorted(cum, target) + 1)
    else:
        if max_rank is None:
            k = len(S)
        else:
            k = min(max_rank, len(S))

    U_k = U[:, :k]
    S_k = S[:k]
    V_k = Vh[:k, :].T

    # absorb Σ into the left factor
    U_fac = U_k * S_k[np.newaxis, :]
    V_fac = V_k

    A_k = U_fac @ V_fac.T
    rel_error = np.linalg.norm(A - A_k, ord="fro") / np.linalg.norm(A, ord="fro")

    return U_fac, V_fac, rel_error


# ---------------------------------------------------------------------------
# 3. ACA with complete pivoting
# ---------------------------------------------------------------------------

def aca_complete_pivoting(A, epsilon=1e-10, max_rank=None):
    """
    Adaptive Cross Approximation (ACA) with complete pivoting.

    At each iteration k:
        - Find global pivot (i_k, j_k) that maximizes |R_k(i, j)|
          where R_k is the current residual.
        - Set u_k = R_k[:, j_k], v_k = R_k[i_k, :] / R_k[i_k, j_k].
        - Update R_{k+1} = R_k - u_k v_k^T.

    This is O(k m n) and uses explicit residual updates, which is fine for TP sizes.

    Parameters
    ----------
    A : (m, n) ndarray
        Matrix to approximate.
    epsilon : float
        Stopping tolerance as a fraction of ||A||_F (if pivot becomes very small).
    max_rank : int or None
        Maximum rank allowed. If None, set max_rank = min(m, n).

    Returns
    -------
    U : (m, k) ndarray
        Left factor.
    V : (n, k) ndarray
        Right factor.
    k : int
        Final rank.
    rel_residual : float
        Relative Frobenius norm of the final residual ||A - U V^T||_F / ||A||_F.
    """
    A = np.array(A, dtype=np.complex128)
    m, n = A.shape

    if max_rank is None:
        max_rank = min(m, n)

    R = A.copy()
    normA = np.linalg.norm(A, 'fro')
    if normA == 0:
        return np.zeros((m, 0), dtype=np.complex128), np.zeros((n, 0), dtype=np.complex128), 0, 0.0

    U_list = []
    V_list = []

    for k in range(max_rank):
        # global pivot
        idx = np.argmax(np.abs(R))
        i_p, j_p = divmod(idx, n)
        pivot = R[i_p, j_p]

        if np.abs(pivot) <= epsilon * normA:
            break

        u_k = R[:, j_p].copy()
        v_k = R[i_p, :].copy() / pivot

        U_list.append(u_k)
        V_list.append(v_k)

        # update residual
        R -= np.outer(u_k, v_k)

    if not U_list:
        U = np.zeros((m, 0), dtype=np.complex128)
        V = np.zeros((n, 0), dtype=np.complex128)
        rel_residual = 1.0
        final_rank = 0
    else:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        final_rank = U.shape[1]
        R_final = A - U @ V.T
        rel_residual = np.linalg.norm(R_final, 'fro') / normA

    return U, V, final_rank, rel_residual


# ---------------------------------------------------------------------------
# 4. ACA with partial pivoting
# ---------------------------------------------------------------------------

def aca_partial_pivoting(A, epsilon=1e-10, max_rank=None):
    """
    ACA with *row/column* partial pivoting.

    This variant avoids scanning the *whole* residual at each step:

      - Choose a row index i_k (first one via row norm, then via u_{k-1}).
      - In that row, pick column j_k with maximal residual |r(i_k, j)|.
      - In that column, build the residual column c.
      - Use (i_k, j_k) as pivot, construct rank-one update u_k, v_k.
      - Next row index chosen from |u_k|.

    NOTES:
      * For "nice" kernel matrices (like many BEM blocks), this behaves well.
      * For arbitrary random matrices, it can stagnate or pick poor pivots,
        which is precisely one of the pedagogical points of the TP.

    Parameters
    ----------
    A : (m, n) ndarray
        Matrix to approximate.
    epsilon : float
        Stopping tolerance relative to ||A||_F (via pivot size).
    max_rank : int or None
        Maximum rank allowed. If None, set max_rank = min(m, n).

    Returns
    -------
    U : (m, k) ndarray
        Left factor.
    V : (n, k) ndarray
        Right factor.
    k : int
        Final rank used (<= max_rank).
    """
    A = np.array(A, dtype=np.complex128)
    m, n = A.shape

    if max_rank is None:
        max_rank = min(m, n)

    normA = np.linalg.norm(A, 'fro')
    if normA == 0:
        return np.zeros((m, 0), dtype=np.complex128), np.zeros((n, 0), dtype=np.complex128), 0

    U = None
    V = None

    # initial row: one with largest norm
    row_norms = np.linalg.norm(A, axis=1)
    i = int(np.argmax(row_norms))

    k = 0
    while k < max_rank:
        # residual row r(i, :)
        if k == 0:
            r = A[i, :].copy()
        else:
            r = A[i, :] - U[i, :] @ V.T

        j = int(np.argmax(np.abs(r)))

        # residual column c(:, j)
        if k == 0:
            c = A[:, j].copy()
        else:
            c = A[:, j] - U @ V[j, :].conj().T

        pivot = c[i]
        if np.abs(pivot) <= epsilon * normA:
            break

        u_k = c
        v_k = r / pivot

        if U is None:
            U = u_k[:, None]
            V = v_k[:, None]
        else:
            U = np.column_stack((U, u_k))
            V = np.column_stack((V, v_k))

        k += 1

        # next row index chosen from |u_k|
        i = int(np.argmax(np.abs(u_k)))

    if U is None:
        U = np.zeros((m, 0), dtype=np.complex128)
        V = np.zeros((n, 0), dtype=np.complex128)
        k = 0

    return U, V, k


# ---------------------------------------------------------------------------
# 5. Test routines on synthetic matrices
# ---------------------------------------------------------------------------

def test_on_random_low_rank():
    """
    Run SVD, ACA (complete) and ACA (partial) on a synthetic low-rank matrix
    and print relative errors and execution times as a function of the chosen rank.
    """
    print("\n=== Test on synthetic low-rank matrix ===")

    m, n, r_true = 80, 50, 5
    noise_level = 1e-8

    A, _, _ = generate_low_rank_matrix(m, n, r_true, noise_level=noise_level,
                                       complex_valued=False, seed=42)
    normA = np.linalg.norm(A, 'fro')
    print(f"Matrix size: {m} x {n}, target rank: {r_true}, ||A||_F = {normA:.4e}")

    ranks_to_test = [1, 2, 3, 4, 5, 10, 15]

    # --- 1. Truncated SVD ---
    print("\n--- Truncated SVD ---")
    for k in ranks_to_test:
        t0 = time.perf_counter()
        U_fac, V_fac, rel_err = truncated_svd(A, tol=None, max_rank=k)
        t1 = time.perf_counter()
        print(f"  k={k:2d} | rel error: {rel_err:.4e} | time: {t1 - t0:.4e} s")

    # --- 2. ACA with complete pivoting ---
    print("\n--- ACA with complete pivoting ---")
    for k in ranks_to_test:
        t0 = time.perf_counter()
        U_aca, V_aca, k_used, rel_res = aca_complete_pivoting(A, epsilon=1e-15, max_rank=k)
        t1 = time.perf_counter()
        print(f"  max_rank={k:2d} | used k={k_used:2d} | rel residual: {rel_res:.4e} | time: {t1 - t0:.4e} s")

    # --- 3. ACA with partial pivoting ---
    print("\n--- ACA with partial pivoting ---")
    for k in ranks_to_test:
        t0 = time.perf_counter()
        U_p, V_p, k_used = aca_partial_pivoting(A, epsilon=1e-15, max_rank=k)
        t1 = time.perf_counter()
        A_k = U_p @ V_p.T
        rel_err = np.linalg.norm(A - A_k, 'fro') / normA
        print(f"  max_rank={k:2d} | used k={k_used:2d} | rel error: {rel_err:.4e} | time: {t1 - t0:.4e} s")

    print("\n(Observation: on arbitrary random matrices, partial-pivoting ACA "
          "may behave worse than complete pivoting; that's expected.)")


# ---------------------------------------------------------------------------
# 6. Application to a BEM matrix block (TP 1–2 solver)
# ---------------------------------------------------------------------------

def test_on_bem_block():
    """
    Build the BEM matrix from solver_BEM.py for a circle and apply
    SVD + ACA on one off-diagonal (far-field) block A[I, J].
    """
    try:
        from solver_BEM import M, u_inc, gamma 
        from Mesh_generator import build_mesh_from_points
    except ImportError:
        print("solver_BEM.py and Mesh_generator.py must be importable for this test.")
        return

    print("\n=== Test on a BEM block ===")

    # Basic BEM configuration: circular obstacle, same as TP1/TP2
    k = np.pi
    a = 1.0
    n_points = 200
    n_quad = 2

    theta = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    boundary_pts = a * np.column_stack([np.cos(theta), np.sin(theta)])

    mesh = build_mesh_from_points(boundary_pts, closed=True)

    print(f"Assembling BEM matrix M (n={n_points}) ...")
    t0 = time.perf_counter()
    A = M(mesh, n_quad, n_points, k)
    t1 = time.perf_counter()
    print(f"Assembly done in {t1 - t0:.3f} s")

    # Choose two well-separated index clusters I, J
    # e.g. quarter arcs on opposite sides of the circle
    I = np.arange(0, n_points // 4)
    J = np.arange(n_points // 2, n_points // 2 + n_points // 4)
    A_block = A[np.ix_(I, J)]

    m, n = A_block.shape
    norm_block = np.linalg.norm(A_block, 'fro')
    print(f"Block size: {m} x {n}, ||A_block||_F = {norm_block:.4e}")

    # Numerical rank via SVD (with tolerance on singular values)
    U_full, S_full, Vh_full = np.linalg.svd(A_block, full_matrices=False)
    s_rel = S_full / S_full[0]
    num_rank = np.sum(s_rel > 1e-6)
    print(f"  Numerical rank (SVD, tol=1e-6 on σ/σ_max): {num_rank}")

    tol_list = [1e-1, 1e-2, 1e-3]

    # --- SVD-based approximation on the BEM block ---
    print("\n--- Truncated SVD on BEM block ---")
    for tol in tol_list:
        t0 = time.perf_counter()
        U_fac, V_fac, rel_err = truncated_svd(A_block, tol=tol, max_rank=None)
        t1 = time.perf_counter()
        r = U_fac.shape[1]
        print(f"  tol={tol:.0e} | rank={r:3d} | rel error={rel_err:.4e} | time={t1 - t0:.4e} s")

    # --- ACA complete pivoting on the BEM block ---
    print("\n--- ACA (complete pivoting) on BEM block ---")
    for tol in tol_list:
        t0 = time.perf_counter()
        U_aca, V_aca, r_aca, rel_res = aca_complete_pivoting(A_block, epsilon=tol, max_rank=None)
        t1 = time.perf_counter()
        print(f"  epsilon={tol:.0e} | rank={r_aca:3d} | rel residual={rel_res:.4e} | time={t1 - t0:.4e} s")

    # --- ACA partial pivoting on the BEM block ---
    print("\n--- ACA (partial pivoting) on BEM block ---")
    for tol in tol_list:
        t0 = time.perf_counter()
        U_p, V_p, r_p = aca_partial_pivoting(A_block, epsilon=tol, max_rank=None)
        t1 = time.perf_counter()
        A_k = U_p @ V_p.T
        rel_err = np.linalg.norm(A_block - A_k, 'fro') / norm_block
        print(f"  epsilon={tol:.0e} | rank={r_p:3d} | rel error={rel_err:.4e} | time={t1 - t0:.4e} s")

    print("\nObservation you’ll want to comment in the report:")
    print("- SVD gives the best accuracy for a given rank but is the most expensive.")
    print("- ACA with complete pivoting is cheaper than full SVD but still scans the full matrix.")
    print("- ACA with partial pivoting can be cheaper but may be less robust;")
    print("  on structured BEM blocks, its behavior is usually better than on random matrices.")

if __name__ == "__main__":
    # Synthetic tests
    test_on_random_low_rank()

    # BEM block test
    test_on_bem_block()
