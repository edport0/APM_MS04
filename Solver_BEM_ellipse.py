import numpy as np
import matplotlib.pyplot as plt

from quad_gl import quad_segment
from Mesh_generator import build_mesh_from_points
from scipy.special import hankel1 as H1
from compute_u_scat import calculate_u_scat_integral

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
def b(k,mesh,n_quad,n_points):
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
    #Parâmetros 
    k =  np.pi    # Número de onda
    n_quad = 7      # Ordem da quadratura de Gauss
    theta0=0*np.pi/4 #angulo de incidencia

    #Boundary mesh
    a=1 #semieixo maior
    c=.1 #semieixo menor
    n_points=200 #pontos na fronteira

    theta=np.linspace(0,2*np.pi,n_points, endpoint=False)
    r=(a*c)/np.sqrt((c*np.cos(theta))**2+(a*np.sin(theta))**2)
    boundary_points=np.array([r*np.cos(theta),r*np.sin(theta)]).T
    mesh=build_mesh_from_points(boundary_points,closed=True)

    #plot boundary mesh
    cyc = np.vstack([mesh.points, mesh.points[0]])
    plt.plot(cyc[:,0], cyc[:,1], 'o-', label="boundary points")
    for seg in mesh.extremities:
        x, y = seg[:,0], seg[:,1]
        plt.plot(x, y, 'k-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(); 
    plt.show()

    #compute trace
    B = -b(k,mesh,n_quad,n_points) 
    A=M(mesh,n_quad,n_points,k)
    p = np.linalg.solve(A, B) 

    # plot segundo membro
    plt.plot(np.real(B), label='real', color='blue')
    plt.plot(np.imag(B), label='imag', color='green')
    plt.title(f'Seconde terme')
    plt.legend()
    plt.show()

    # plot do traço
    plt.plot(theta, np.real(p), color='blue', label='real')
    plt.plot(theta, np.imag(p), color='black', label='imag')
    plt.title('k = {:.2f}'.format(k))
    plt.legend()
    plt.show()

    #calcul du volume

    # --- 1. Définir une grille cartésienne dans le volume ---
    R_box = 3 * a 
    n_grid_pts = 200 # Résolution de l'image (200x200 pixels)

    # Vecteurs pour les axes x et y
    x_vec = np.linspace(-R_box, R_box, n_grid_pts)
    y_vec = np.linspace(-R_box, R_box, n_grid_pts)
    
    # Créer les matrices 2D pour X et Y
    X_vol, Y_vol = np.meshgrid(x_vec, y_vec)

    # --- 2. Calculer le champ incident sur cette grille ---
    u_inc_grid = u_inc_field(X_vol, Y_vol, k, theta0)

    # --- 3. Calculer le champ diffracté (u+) sur la grille ---
    
    # Aplatir la grille 2D en une longue liste de points (xi)
    # Shape: (n_grid_pts * n_grid_pts, 2)
    xi_points = np.vstack([X_vol.ravel(), Y_vol.ravel()]).T
    
    # Initialiser le vecteur solution
    u_scat_field = np.zeros(xi_points.shape[0], dtype=np.complex128)
    
    # Boucle sur chaque point (pixel) de la grille de visualisation
    for i, xi in enumerate(xi_points):
        
        # Vérification: sommes-nous à l'intérieur de l'ellipse?
        # Équation: (x/a)^2 + (y/c)^2 < 1
        if (xi[0]**2 / a**2 + xi[1]**2 / c**2) < 1.0:
            u_scat_field[i] = np.nan # Met np.nan si à l'intérieur
        else:
            # Si à l'extérieur, calculer l'intégrale BEM
            u_scat_field[i] = calculate_u_scat_integral(xi, k, n_quad, mesh, p)

    # --- 4. Remettre en forme et calculer le champ total ---
    u_scat_grid = u_scat_field.reshape(X_vol.shape)

    u_total_grid = u_inc_grid + u_scat_grid

    
    # --- 5. Afficher les trois champs ---

    # Définir une échelle de couleur commune et une map pour la comparabilité
    v_min, v_max = -2, 2
    cmap = 'jet'

    # (Préparation pour dessiner l'ellipse)
    cyc_plot = np.vstack([boundary_points, boundary_points[0]])

    # --- Plot 1: Champ Incident ---
    plt.figure(figsize=(10, 8))
    # Note: On ne masque PAS l'intérieur pour le champ incident
    plt.pcolormesh(X_vol, Y_vol, np.real(u_inc_grid), shading='auto', cmap=cmap, vmin=v_min, vmax=v_max)
    plt.plot(cyc_plot[:, 0], cyc_plot[:, 1], color='black', linewidth=1.5)
    plt.title(f'Champ Incident $Re(u_{'inc'})$, k={k/np.pi:.1f}$pi$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='$Re(u_{inc})$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # --- Plot 2: Champ Diffracté ---
    plt.figure(figsize=(10, 8))
    # u_scat_grid a déjà les NaN à l'intérieur
    plt.pcolormesh(X_vol, Y_vol, np.real(u_scat_grid), shading='auto', cmap=cmap, vmin=v_min, vmax=v_max)
    plt.plot(cyc_plot[:, 0], cyc_plot[:, 1], color='black', linewidth=1.5)
    plt.title(f'Champ Diffracté $Re(u_s)$, k={k/np.pi:.1f}$pi$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='$Re(u_s)$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # --- Plot 3: Champ Total ---
    plt.figure(figsize=(10, 8))
    # u_total_grid a déjà les NaN à l'intérieur
    plt.pcolormesh(X_vol, Y_vol, np.real(u_total_grid), shading='auto', cmap=cmap, vmin=v_min, vmax=v_max)
    plt.plot(cyc_plot[:, 0], cyc_plot[:, 1], color='black', linewidth=1.5)
    plt.title(f'Champ Total $Re(u_{'total'})$, k={k/np.pi:.1f}$pi$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='$Re(u_{total})$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()