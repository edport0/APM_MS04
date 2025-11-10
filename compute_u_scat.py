# Calcula o campo difratado u⁺ em um ponto de observação usando a representação integral.
# Valida o resultado contra a solução analítica em série do TP0.

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv as J, hankel1 as H1
from matplotlib.patches import Circle

# --- Importações dos seus módulos ---
from Mesh_generator import build_mesh_from_points # Importa a função do seu arquivo
from compute_trace_analy import dr_u_inc, dr_u_scattered # Funções para q_inc, q_scat e u⁺ analítica
from quad_gl import quad_segment

def green_function(x_obs, y_src, k):
    """Calcula a Função de Green G(x, y) = i/4 * H0(k*||x-y||)."""
    # x_obs: ponto de observação (shape (2,))
    # y_src: pontos fonte (shape (q, 2))
    dist_vectors = y_src - x_obs[None, :]
    dist = np.linalg.norm(dist_vectors, axis=1)
    
    return (1j / 4) * H1(0, k * dist)

def trace_analy(k,a,theta_col,theta0,N):
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

def u_inc_field(X, Y, k, theta0=0.0):
    """
    Calcula o campo da onda plana incidente u_inc = exp(-i*k*(x*cos(theta0) + y*sin(theta0)))
    em uma grade de pontos cartesianos (X, Y).
    """
    # Rotação para a direção da onda incidente
    x_rotated = X * np.cos(theta0) + Y * np.sin(theta0)
    return np.exp(-1j * k * x_rotated)

def calculate_u_scat_integral(x_obs, k, n_quad, mesh, p_values):
    
    total_integral =0.0+ 0.0j
    
    for i in range(mesh.segments.shape[0]): # mesh.segments.shape[0] é o número de segmentos
        y0 = mesh.extremities[i, 0, :] 
        y1 = mesh.extremities[i, 1, :] 
        
        p_j = p_values[i] 

        integrand_j = lambda Y_src: green_function(x_obs, Y_src, k) * p_j
        
        segment_integral = quad_segment(integrand_j, y0, y1, n_quad)
        
        total_integral += segment_integral
        
    return total_integral


if __name__ == "__main__":
#   Parâmetros 
    k = np.pi         # Número de onda
    a = 1.0          # Raio do disco
    n_points = 200  # Número de pontos na fronteira
    n_quad = 4      # Ordem da quadratura de Gauss
    N_fourier = 40   # Número de modos para as séries
    theta0=0.0      #incidence direction

#discretização do campo 
    box_size=5*a
    n_radial=80
    n_angular=200

# ponto do calculo do erro
    xi_vld = np.array([0, 2]) # Ponto de observação

# Disk mesh
    theta_boundary = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    boundary_pts = a * np.array([np.cos(theta_boundary), np.sin(theta_boundary)]).T
    mesh = build_mesh_from_points(boundary_pts, closed=True)
    midpoints = 0.5 * (mesh.extremities[:, 0, :] + mesh.extremities[:, 1, :])
    theta_col = np.arctan2(midpoints[:, 1], midpoints[:, 0])

# compute trace
    p_values = -trace_analy(k,a,theta_col,theta0,N=N_fourier)

# mesh for the fields
    epsilon = 1e-5
    r_coords = np.linspace(a+epsilon, box_size, n_radial)
    theta_coords = np.linspace(0, 2 * np.pi, n_angular)
    
    R, Theta = np.meshgrid(r_coords, theta_coords)
    X_grid = R * np.cos(Theta)
    Y_grid = R * np.sin(Theta)

# compute u_inc
    u_inc_grid = u_inc_field(X_grid, Y_grid, k, theta0)

# compute u+
    # Converte a grade em uma lista de pontos xi (shape: [discretization*discretization, 2])
    xi_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    u_scat_field = np.zeros(xi_points.shape[0], dtype=np.complex128)
    
    # Loop sobre cada ponto de observação xi
    for i, xi in enumerate(xi_points):
        # Ignora pontos dentro do disco
        if np.linalg.norm(xi) < a:
            u_scat_field[i] = np.nan # Marca como inválido (Not a Number)
            continue
            
        # Reutiliza a malha e a densidade p já calculadas
        u_scat_field[i] = calculate_u_scat_integral(xi, k, n_quad, mesh, p_values)

    # Remodela o vetor de resultados de volta para o formato de grade 2D
    u_scat_grid = u_scat_field.reshape((R.shape))

# compute u tot
    u_total_grid = u_inc_grid + u_scat_grid
    u_total_grid[R < a] = np.nan

# calculo do erro relativo em xi_vld
    u_scat_numerical = calculate_u_scat_integral(xi_vld, k, n_quad, mesh, p_values)
    
    r_obs = np.linalg.norm(xi_vld)
    theta_obs = np.arctan2(xi_vld[1], xi_vld[0])
    print(r_obs,theta_obs)
    u_scat_exact = u_scat_analytical(r_obs, theta_obs, k, a, N_fourier)
    
    print(f"Solução Numérica (via Integral): u⁺(xi) = {u_scat_numerical}")
    print(f"Solução Analítica (via Série):   u⁺(xi) = {u_scat_exact}")
    
    error = np.abs(u_scat_numerical - u_scat_exact) / np.abs(u_scat_exact)
    print(f"\nErro Relativo: {error:.4e}")

# vérifier la condition de dirichlet

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

# --- Visualização dos Três Campos ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': 'polar'})
    #fig.suptitle(f"Análise de Difração de Onda (k={k}, a={a})", fontsize=16)
    
    # Encontrar limites de cor consistentes para u_inc e u_total
    vmax = np.max(np.abs(np.real(u_inc_grid)))
    vmin = -vmax

    # Plot 1: Campo Incidente
    axes[0].pcolormesh(Theta, R, np.real(u_inc_grid), cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("u_inc")
    axes[0].set_yticklabels([])

    # Plot 2: Campo Espalhado
    axes[1].pcolormesh(Theta, R, np.real(u_scat_grid), cmap='jet', shading='auto')
    axes[1].set_title("u⁺")
    axes[1].set_yticklabels([])

    # Plot 3: Campo Total
    im = axes[2].pcolormesh(Theta, R, np.real(u_total_grid), cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title("u_total")
    axes[2].set_yticklabels([])
    
    # Adicionar o disco e a colorbar
    for ax in axes:
        ax.fill_between(np.linspace(0, 2*np.pi, 100), 0, a, color='white', zorder=10)
        ax.set_yticklabels([]) 
        ax.set_thetagrids([])

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Re(u)", shrink=0.6)
    
    plt.show()

# Plot radial
    theta_plot = 0*np.pi
    theta_index = np.argmin(np.abs(theta_coords - theta_plot))
    r_axis = r_coords
    
    #u_inc_radial = u_inc_grid[theta_index, :]
    u_scat_radial = u_scat_grid[theta_index, :]
    #u_total_radial = u_total_grid[theta_index, :]
    u_scat_analytical_radial = np.array([u_scat_analytical(r, theta_plot, k, a, N_fourier) for r in r_axis])

    plt.figure(figsize=(12, 7))
    
# Plot da parte real de cada campo
    plt.plot(r_axis, np.real(u_scat_radial), label=r'$Re(u^{+})$', linestyle='solid', color='blue')
    plt.plot(r_axis, np.imag(u_scat_radial), label=r'$Im(u^{+})$', linestyle='solid', color='red')
    plt.plot(r_axis, np.real(u_scat_analytical_radial), label=r'$Re(u^{+}_{analy})$', linestyle='--', color='yellow', linewidth=1.5)
    plt.plot(r_axis, np.imag(u_scat_analytical_radial), label=r'$Im(u^{+}_{analy})$', linestyle='--', color='black', linewidth=1.5)
    
    plt.xlabel("Rayon (r)")
    plt.title(f"Champs diffractés pour θ = {np.degrees(theta_plot):.1f}°")
    plt.legend()
    plt.xlim(a, box_size)
    plt.show()