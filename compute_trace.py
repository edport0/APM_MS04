# compute_trace.py
"""
Compute ∂_ν u^{inc}, ∂_ν u^+ and ∂_ν u^{tot} on a disk boundary mesh
for the 2D Helmholtz Dirichlet scattering problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from Mesh_generator import build_mesh_from_points   # your mesh builder

# --- Bessel/Hankel helpers ---
from scipy.special import jv, yv

def H1(n, z):   # Hankel H_n^(1)
    return jv(n, z) + 1j * yv(n, z)

def J(n, z):    # Bessel J_n
    return jv(n, z)

# --- Traces ---

def dr_u_inc(k, a, theta, theta0=0.0, N=50):
    """Radial derivative of incident plane wave on r=a."""
    n_all = np.arange(-N, N+1)
    Jm1 = np.array([J(n-1, k*a) for n in n_all])
    Jp1 = np.array([J(n+1, k*a) for n in n_all])
    coeff = (-1j)**n_all * (k/2) * (Jm1 - Jp1)
    e_inθ = np.exp(1j * n_all[:,None] * (theta[None,:] - theta0))
    return np.sum(coeff[:,None] * e_inθ, axis=0)

def dr_u_scattered(k, a, theta, N=50):
    """Radial derivative of scattered field on r=a."""
    n_all = np.arange(-N, N+1)
    Hm1 = np.array([H1(n-1, k*a) for n in n_all])
    Hp1 = np.array([H1(n+1, k*a) for n in n_all])
    Hn  = np.array([H1(n,   k*a) for n in n_all])
    Jn  = np.array([J(n,   k*a) for n in n_all])

    coeff = -( (-1j)**n_all * Jn / Hn ) * (k/2)*(Hm1 - Hp1)
    e_inθ = np.exp(1j * n_all[:,None] * theta[None,:])
    return np.sum(coeff[:,None] * e_inθ, axis=0)

# --- Main driver ---
if __name__ == "__main__":
    # Parameters
    k = 5.0        # wavenumber
    a = 1.0        # disk radius
    n_points = 80  # boundary discretization
    N = 40         # Fourier truncation
    theta0 = 0.0   # incidence direction (plane wave along +x)

    # Build disk mesh
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    pts = np.column_stack((a*np.cos(t), a*np.sin(t)))
    mesh = build_mesh_from_points(pts, closed=True)

    # Collocation angles = angles of segment midpoints
    mids = 0.5*(mesh.extremities[:,0] + mesh.extremities[:,1])
    theta_col = np.arctan2(mids[:,1], mids[:,0])

    # Compute traces
    dr_inc = dr_u_inc(k, a, theta_col, theta0=theta0, N=N)
    dr_scat = dr_u_scattered(k, a, theta_col, N=N)
    dr_tot = dr_inc + dr_scat

    # Print a sample
    print("First few collocation points (θ):", theta_col[:5])
    print("∂r u^inc (first 5):", dr_inc[:5])
    print("∂r u^+   (first 5):", dr_scat[:5])
    print("∂r u^tot (first 5):", dr_tot[:5])

    order = np.argsort(theta_col)                    # sort angles
    ths   = theta_col[order]
    vals  = np.abs(dr_tot)[order]

    #Plot mesh
    cyc = np.vstack([mesh.points, mesh.points[0]])
    plt.plot(cyc[:,0], cyc[:,1], 'o-', label="boundary points")
    for seg in mesh.extremities:
        x, y = seg[:,0], seg[:,1]
        plt.plot(x, y, 'k-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(); plt.title("Disk boundary mesh")
    plt.show()

    # Plot magnitude of total trace vs θ
    plt.figure()
    plt.plot(ths, vals, 'o-')
    plt.xlabel("θ [rad]")
    plt.ylabel("|∂ν u^tot|")
    plt.title("Trace of total radial derivative on Γ")
    plt.show()
