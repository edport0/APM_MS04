# main_tp1.py
# Calcula o campo difratado u⁺ em um ponto de observação usando a representação integral.
# Valida o resultado contra a solução analítica em série do TP0.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv as J, hankel1 as H1

# --- Importações dos seus módulos ---
from Mesh_generator import build_mesh_from_points # Importa a função do seu arquivo
from compute_trace import dr_u_inc, dr_u_scattered # Funções para q_inc, q_scat e u⁺ analítica
from quad_gl import quad_segment

def green_function(x_obs, y_src, k):
    """Calcula a Função de Green G(x, y) = i/4 * H0(k*||x-y||)."""
    # x_obs: ponto de observação (shape (2,))
    # y_src: pontos fonte (shape (q, 2))

    dist_vectors = y_src - x_obs[None, :]
    dist = np.linalg.norm(dist_vectors, axis=1)
    
    return (1j / 4) * H1(0, k * dist)

#def p_density(Y_segments, k, a, N_fourier):
    """
    Calcula a densidade p(y) = (q_inc(y) + q_scat(y))
    para um conjunto de pontos Y_segments, que são os pontos médios dos segmentos.
    """
    # Y_segments é uma matriz (num_segments, 2) contendo os pontos médios
    theta_segments = np.arctan2(Y_segments[:, 1], Y_segments[:, 0])
    q_inc = dr_u_inc(k, a, theta_segments, N_fourier)
    q_scat = dr_u_scattered(k, a, theta_segments, N_fourier)

    return (q_inc + q_scat)

def trace(k,a,theta_col,theta0,N):
    dr_inc = dr_u_inc(k, a, theta_col, theta0=theta0, N=N)
    dr_scat = dr_u_scattered(k, a, theta_col, N=N)
    return (dr_inc + dr_scat)

def u_scat_analytical(r_obs, theta_obs, k, a, N):
    n_all=np.arange(-N,N+1)
    Jn_a = J(n_all, k * a)
    Hn_a = H1(n_all, k * a)
    Hn = H1(n_all, k * r_obs)
    exp = np.exp(1j*n_all*theta_obs) 
    coeff = -(-1j)**n_all*(Jn_a/Hn_a)

    return np.sum(coeff*Hn*exp)

def calculate_u_plus_integral(x_obs, k, n_quad, mesh, p_values):
    
    total_integral = 0.0j
    
    print(f"Integrando sobre {mesh.segments.shape[0]} segmentos...")
    for i in range(mesh.segments.shape[0]): # mesh.segments.shape[0] é o número de segmentos
        y0 = mesh.extremities[i, 0, :] 
        y1 = mesh.extremities[i, 1, :] 
        
        p_j = p_values[i] 

        integrand_j = lambda Y_src: green_function(x_obs, Y_src, k) * p_j
        
        segment_integral = quad_segment(integrand_j, y0, y1, n_quad)
        
        total_integral += segment_integral
        
    return total_integral


if __name__ == "__main__":
    # --- Parâmetros ---
    k = 1.0          # Número de onda
    a = 1.0          # Raio do disco
    n_points = 10  # Número de pontos da malha (também o número de segmentos para closed=True)
    n_quad = 4      # Ordem da quadratura de Gauss
    N_fourier = 40   # Número de modos para as séries
    theta0=0.0      #incidence direction
    xi = np.array([2.0, 0]) # Ponto de observação

    print("--- TP1: Implementação da Representação Integral ---")
    print(f"Calculando u⁺ em xi = {xi}...")

    theta_boundary = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    boundary_pts = a * np.array([np.cos(theta_boundary), np.sin(theta_boundary)]).T
    mesh = build_mesh_from_points(boundary_pts, closed=True)

    midpoints = 0.5 * (mesh.extremities[:, 0, :] + mesh.extremities[:, 1, :])
    theta_col = np.arctan2(midpoints[:, 1], midpoints[:, 0])

    p_values = trace(k,a,theta_col,theta0,N=N_fourier)

    u_plus_numerical = calculate_u_plus_integral(xi, k, n_quad, mesh, p_values)
    
    r_obs = np.linalg.norm(xi)
    theta_obs = np.arctan2(xi[1], xi[0])
    print(r_obs,theta_obs)
    u_plus_exact = u_scat_analytical(r_obs, theta_obs, k, a, N_fourier)
    
    # --- Resultados ---
    print(f"Solução Numérica (via Integral): u⁺(xi) = {u_plus_numerical}")
    print(f"Solução Analítica (via Série):   u⁺(xi) = {u_plus_exact}")
    
    error = np.abs(u_plus_numerical - u_plus_exact) / np.abs(u_plus_exact)
    print(f"\nErro Relativo: {error:.4e}")

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
    order = np.argsort(theta_col)                    # sort angles
    ths   = theta_col[order]
    vals  = np.abs(p_values)[order]

    plt.figure()
    plt.plot(ths, vals, 'o-')
    plt.xlabel("θ [rad]")
    plt.ylabel("|∂ν p|")
    plt.title("Trace of total radial derivative on Γ")
    plt.show()