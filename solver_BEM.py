import numpy as np
import matplotlib.pyplot as plt

from quad_gl import quad_segment
from Mesh_generator import build_mesh_from_points
from scipy.special import hankel1 as H1
from compute_u_scat import calculate_u_scat_integral, trace_analy

gamma = 0.5772156649 #constant d'Euler

#calcula u_inc em um ponto do R^2 : (x,y)
def u_inc (k, X, theta0=0.0):
    x_inc = X[:,0]*np.cos(theta0)+X[:,1]*np.sin(theta0)

    return np.exp(-1j*k*x_inc)

#calcula G(z1,z2) para z1=(x1,y1), z2=(x2,y2) dois pontos de R^2
def green(z1, z2, k):
    dist_vectors = z2 - z1[None, :]
    dist = np.linalg.norm(dist_vectors, axis=1)
    
    return (1j / 4) * H1(0, k * dist) #para elementos extradiagonais

def green_singular(z_i, z_j_array):
    dist = np.linalg.norm(z_j_array - z_i[None, :], axis=1)
    # Adicionamos uma pequena constante para evitar log(0) se um ponto coincidir,
    epsilon = 1e-15 

    return (-1 / (2 * np.pi)) * np.log(dist + epsilon)

def green_regular(z_i, z_j_array, k):
    dist = np.linalg.norm(z_j_array - z_i[None, :], axis=1)

    const_part = (1j / 4) + (1 / (2 * np.pi)) * (np.log(k / 2) + gamma)
    regular_part = np.full(dist.shape, const_part, dtype=np.complex128)

    # Define um limiar para decidir quando a subtração é segura
    threshold = 1e-10
    
    # Encontra os índices onde a distância é grande o suficiente
    safe_indices = np.where(dist > threshold)[0]

    # Apenas para esses índices, calcula a G_reg pela subtração
    if safe_indices.size > 0:
        z_j_safe = z_j_array[safe_indices]
        regular_part[safe_indices] = green(z_i, z_j_safe, k) - green_singular(z_i, z_j_safe)

    return regular_part

#calcula o segundo membro
def b (k,mesh,n_quad,n_points):
    b=np.zeros(n_points, dtype=np.complex128)
    for i in range(n_points): 
        Y0 = mesh.extremities[i,0,:] #size (n_points, [x1,x2])
        Y1 = mesh.extremities[i,1,:]
        g = lambda X: u_inc(k,X,theta0=0.0)
        b[i] = quad_segment(g,Y0,Y1,n_quad)

    return b

def calculate_Mii_regular(Y0i, Y1i, k, n_quad):
    # Define a função para a integral interna: F_reg(z_i) = ∫_Γi G_reg(z_i, z_j) dΓj
    def inner_integral_regular(z_i_array):
        results = np.zeros(z_i_array.shape[0], dtype=np.complex128)
        for idx, z_i_atual in enumerate(z_i_array):
            integrand = lambda z_j_array: green_regular(z_i_atual, z_j_array, k)
            results[idx] = quad_segment(integrand, Y0i, Y1i, n_quad)
        return results

    # Calcula a integral externa: M_ii^reg = ∫_Γi F_reg(z_i) dΓi
    return quad_segment(inner_integral_regular, Y0i, Y1i, n_quad)

def calculate_Mii_singular(Y0i, Y1i, n_quad):
    segment_vector = Y1i - Y0i
    L = np.linalg.norm(segment_vector) # Comprimento do segmento |Γ_i|
    tau = segment_vector / L             # Vetor tangente unitário

    # Função que calcula a integral interna analiticamente para um array de pontos z_i
    def inner_integral_singular_analytical(z_i_array):
        d_j = Y0i[None, :] - z_i_array
        d_j_plus_1 = Y1i[None, :] - z_i_array

        # Projeções sobre o vetor tangente tau
        d_j_dot_tau = np.einsum('ij,j->i', d_j, tau)
        d_j_plus_1_dot_tau = np.einsum('ij,j->i', d_j_plus_1, tau)

        # Normas dos vetores
        norm_d_j = np.linalg.norm(d_j, axis=1)
        norm_d_j_plus_1 = np.linalg.norm(d_j_plus_1, axis=1)
        
        epsilon = 1e-15 # Evita log(0) nos endpoints

        # Implementação da Equação (5)
        term1 = d_j_plus_1_dot_tau * np.log(norm_d_j_plus_1 + epsilon)
        term2 = d_j_dot_tau * np.log(norm_d_j + epsilon)
        
        return (-1 / (2 * np.pi)) * (term1 - term2 - L)

    # A integral externa é calculada numericamente usando a fórmula analítica da interna
    return quad_segment(inner_integral_singular_analytical, Y0i, Y1i, n_quad)

#calculo cabuloso da matriz
def M (mesh,n_quad,n,k):
    M_matrix=np.zeros((n,n), dtype=np.complex128)
    for i in range(n):
        Y0i=mesh.extremities[i,0,:]
        Y1i=mesh.extremities[i,1,:]

        for j in range(n):
            Y0j=mesh.extremities[j,0,:]
            Y1j=mesh.extremities[j,1,:]

            if (i != j):
                def integral_elemento_j(pontos_elemento_i):
                    results = np.zeros(pontos_elemento_i.shape[0], dtype=np.complex128) #initialize
        
                    for idx, zi in enumerate(pontos_elemento_i): #idx é o contador de posição
                        
                        integrando = lambda pontos_elemento_j: green(zi, pontos_elemento_j, k)
                        results[idx] = quad_segment(integrando, Y0j, Y1j, n_quad)
                        
                    return results

                M_matrix[i, j] = quad_segment(integral_elemento_j, Y0i, Y1i, n_quad)
            
            else :
                M_ii_reg = calculate_Mii_regular(Y0i, Y1i, k, n_quad)
                M_ii_sing = calculate_Mii_singular(Y0i, Y1i, n_quad)
                M_matrix[i, i] = M_ii_reg + M_ii_sing

    return M_matrix

def u_inc_field(X, Y, k, theta0=0.0):
    x_rotated = X * np.cos(theta0) + Y * np.sin(theta0)

    return np.exp(-1j * k * x_rotated)

if __name__ == "__main__":
#   Parâmetros 
    k =  np.pi    # Número de onda
    a = 1.0          # Raio do disco
    n_points = 200  # Número de pontos na fronteira
    n_quad = 7      # Ordem da quadratura de Gauss
    box_size=5*a    #tamanho do volume
    n_radial, n_angular = 80, 200 # malha no volume
    theta0=0.0 #angulo de incidencia

    # boundary mesh
    theta_boundary = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    boundary_pts = a * np.array([np.cos(theta_boundary), np.sin(theta_boundary)]).T
    mesh = build_mesh_from_points(boundary_pts, closed=True)
    mids = 0.5*(mesh.extremities[:,0] + mesh.extremities[:,1])
    theta_col = np.arctan2(mids[:,1], mids[:,0])

    # calcul de la trace
    B = -b(k,mesh,n_quad,n_points) 
    A=M(mesh,n_quad,n_points,k)
    p = np.linalg.solve(A, B) #numerico
    #p_analy = -trace_analy(k,a,theta_col,theta0,N=40)

# plot segundo membro
    plt.plot(np.real(B), label='real', color='blue')
    plt.plot(np.imag(B), label='imag', color='green')
    plt.title(f'Seconde terme')
    plt.legend()
    plt.show()

    # plot do traço
    #plt.plot(theta_boundary, np.abs(p), color='black', label='abs')
    plt.plot(theta_boundary, np.imag(p), color='black', label='numérique')
    plt.title('k = {:.2f}'.format(k))
    plt.legend()
    plt.show()

    # maillage dans le volume
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
        u_scat_field[i] = calculate_u_scat_integral(xi, k, n_quad, mesh, p)

    # Remodela o vetor de resultados de volta para o formato de grade 2D
    u_scat_grid = u_scat_field.reshape((R.shape))

    # compute u tot
    u_total_grid = u_inc_grid + u_scat_grid
    u_total_grid[R < a] = np.nan

    # plot dos três campos ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': 'polar'})
    
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




