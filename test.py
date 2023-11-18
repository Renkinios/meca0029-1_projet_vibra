import matplotlib.pyplot as plt

# Données à tracer
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 5]

# Tracer la première ligne en pointillés et couleur bleue
plt.plot(x, y, linestyle='--', label='Ligne en pointillés')

# Tracer la deuxième ligne avec une couleur plus légère en ajustant l'alpha
plt.plot(x, y, linestyle='-', alpha=0.7, label='Ligne solide légère')

# Ajouter des labels et une légende
plt.xlabel('Axe des X')
plt.ylabel('Axe des Y')
plt.title('Deux lignes superposées')
plt.legend()

# Afficher le graphique
plt.show()
