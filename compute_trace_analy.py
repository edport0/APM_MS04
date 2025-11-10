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

N_fourier = 40

# --- Traces ---

def dr_u_inc(k, a, theta, theta0=0.0, N=N_fourier):
    """Radial derivative of incident plane wave on r=a."""
    n_all = np.arange(-N, N+1) #vector [-N,N]
    Jm1 = np.array([J(n-1, k*a) for n in n_all])
    Jp1 = np.array([J(n+1, k*a) for n in n_all])
    coeff = (-1j)**n_all * (k/2) * (Jm1 - Jp1)
    e_inθ = np.exp(1j * n_all[:,None] * (theta[None,:] - theta0))
    return np.sum(coeff[:,None] * e_inθ, axis=0)

def dr_u_scattered(k, a, theta, N=N_fourier):
    """Radial derivative of scattered field on r=a."""
    n_all = np.arange(-N, N+1) #vector [-N,N]
    Hm1 = np.array([H1(n-1, k*a) for n in n_all])
    Hp1 = np.array([H1(n+1, k*a) for n in n_all])
    Hn  = np.array([H1(n,   k*a) for n in n_all])
    Jn  = np.array([J(n,   k*a) for n in n_all])

    coeff = -( (-1j)**n_all * Jn / Hn ) * (k/2)*(Hm1 - Hp1)
    e_inθ = np.exp(1j * n_all[:,None] * theta[None,:])
    return np.sum(coeff[:,None] * e_inθ, axis=0)

def dr_u_validation(k,a,theta):
    sin=np.array(-np.sin(k * a * theta))
    return k * theta * sin

def u_scat_analytical():

   return

# --- Main driver ---
if __name__ == "__main__":
    # Parameters
    k = 1    # wavenumber
    a = 1.0        # disk radius
    n_points = 200  # boundary discretization
    N = N_fourier         # Fourier truncation
    theta0 = 0.0   # incidence direction (plane wave along +x)

    # Build disk mesh
    t = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    pts = np.column_stack((a*np.cos(t), a*np.sin(t)))
    mesh = build_mesh_from_points(pts, closed=True)
    

    # Collocation angles = angles of segment midpoints
    mids = 0.5*(mesh.extremities[:,0] + mesh.extremities[:,1])
    theta_col = np.arctan2(mids[:,1], mids[:,0])

    # Compute traces
    dr_inc = dr_u_inc(k, a, theta_col, theta0=theta0, N=N)
    dr_scat = dr_u_scattered(k, a, theta_col, N=N)
    dr_valid = dr_u_validation(k,a,theta_col)
    dr_tot = -(dr_inc + dr_scat)  

    # Print a sample
    # print("First few collocation points (θ):", theta_col[:5])
    print("∂r u^inc (first 5):", dr_inc[:5])
    print("∂r u^+   (first 5):", dr_scat[:5])
    # print("∂r u^tot (first 5):", dr_tot[:5])

    order = np.argsort(theta_col)                    # sort angles
    ths   = theta_col[order]

    # Extrai e ordena as três componentes de dr_tot
    vals_abs = np.abs(dr_tot)[order]
    vals_real = dr_tot.real[order]
    vals_imag = dr_tot.imag[order]

    # Cria a figura para a plotagem
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plota as três curvas
    ax.plot(ths, vals_abs, color='black', label='abs')
    ax.plot(ths, vals_real, color='blue', label='real')
    ax.plot(ths, vals_imag, color='green', label='imag')
    
    # Configurações do gráfico
    ax.set_title(f"trace analytique \n k={k:.2f}, |N_fourier|={N}")
    ax.set_xlabel(" θ [rad]")
    #ax.set_ylabel("Valor da Derivada")
    ax.set_xlim(-np.pi, np.pi)
    ax.legend()
    
    plt.show()
