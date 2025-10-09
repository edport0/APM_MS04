import numpy as np
import scipy.linalg  # On a besoin de SciPy pour la QR avec pivotage

# Paramètres
m = 5   # Nombre de lignes
n = 5    # Nombre de colonnes
r = 3    # Le rang de la matrice

U_gen = np.random.randn(m, r)
V_gen = np.random.randn(n, r)
A = U_gen @ V_gen.T  # A est de taille (m, n) et de rang r

# --- 2. Trouver les indices des "meilleures" colonnes et lignes ---

# Pour trouver r colonnes indépendantes, on applique QR avec pivotage à A
# La fonction retourne Q, R, et P (indices de permutation)
# Les r premiers indices de P sont ceux des colonnes les plus indépendantes
_ , _ , col_pivots = scipy.linalg.qr(A, pivoting=True) # type: ignore
col_indices = col_pivots[:r]
col_indices.sort() # On les trie pour un affichage plus propre (optionnel)

# Pour trouver r lignes indépendantes, on applique la même logique à la transposée de A (A.T)
_ , _ , row_pivots = scipy.linalg.qr(A.T, pivoting=True) # type: ignore
row_indices = row_pivots[:r]
row_indices.sort() # On les trie aussi

print(f"Indices des {r} colonnes sélectionnées (pour C) : {col_indices}")
print(f"Indices des {r} lignes sélectionnées (pour R)  : {row_indices}")
print("-" * 40)


# --- 3. Construire les matrices C, R et Â ---

# C contient les colonnes sélectionnées de A
# A[:, col_indices] -> prend toutes les lignes (:) et les colonnes spécifiées
C = A[:, col_indices]

# R contient les lignes sélectionnées de A
# A[row_indices, :] -> prend les lignes spécifiées et toutes les colonnes (:)
R = A[row_indices, :]

# Â (A_hat) est l'intersection des lignes et colonnes sélectionnées
# np.ix_ est un outil numpy pratique pour ce type d'indexation croisée
A_hat = A[np.ix_(row_indices, col_indices)]

print(f"Matrice C créée de taille {C.shape}")
print(f"Matrice R créée de taille {R.shape}")
print(f"Matrice Â (A_hat) créée de taille {A_hat.shape}")
print("-" * 40)


# --- 4. Vérification de la décomposition ---

# D'abord, on vérifie que Â est bien inversible
# np.linalg.cond calcule le "conditionnement", un grand nombre indique une quasi-singularité
# Un nombre raisonnable signifie qu'elle est inversible.
cond_A_hat = np.linalg.cond(A_hat)
print(f"Conditionnement de Â : {cond_A_hat:.2f} (un petit nombre est bon)")

# Calcul de l'inverse de Â
A_hat_inv = np.linalg.inv(A_hat)

# Reconstruction de la matrice A à partir de la décomposition
A_reconstructed = C @ A_hat_inv @ R

# Calcul de l'erreur entre la matrice originale et la matrice reconstruite
# La norme de Frobenius est une bonne mesure de la "taille" de l'erreur
error = np.linalg.norm(A - A_reconstructed)

print(f"Norme de l'erreur ||A -C(Â^-1)R|| : {error}")
print("-" * 40)
print(A)
print("-" * 40)
print(A_hat)
print("-" * 40)
print(A_reconstructed)

