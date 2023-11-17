import numpy as np

# Créer deux matrices
matrice1 = np.array([[1, 2], [3, 4]])
matrice2 = np.array([[5, 6]])

# Concaténation horizontale (le long de l'axe des colonnes)
concatenation_horizontale = np.concatenate((matrice1, matrice2.T), axis=1)

# Concaténation verticale (le long de l'axe des lignes)
concatenation_verticale = np.concatenate((matrice1, matrice2), axis=0)

# Afficher les résultats
print("Matrice 1:")
print(matrice1)

print("\nMatrice 2:")
print(matrice2)

print("\nConcaténation horizontale:")
print(concatenation_horizontale)

print("\nConcaténation verticale:")
print(concatenation_verticale)