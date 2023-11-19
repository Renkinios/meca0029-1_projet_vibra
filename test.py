import matplotlib.pyplot as plt

# Données à tracer
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 5]

# Tracer avec différents styles de ligne
plt.plot(x, y, linestyle='-', label='Ligne solide (par défaut)')
plt.plot(x, y, linestyle='--', label='Ligne en pointillés')
plt.plot(x, y, linestyle=':', label='Ligne en points')
plt.plot(x, y, linestyle='-.', label='Ligne en tiret-point')

# Ajouter des labels et une légende
plt.xlabel('Axe des X')
plt.ylabel('Axe des Y')
plt.title('Différents styles de ligne')
plt.legend()

# Afficher le graphique
plt.show()
