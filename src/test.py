import matplotlib.pyplot as plt

# Créer des données de test
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Tracer la ligne constante
plt.plot(x, y)
plt.axhline(y=5, color='r', linestyle='--', label='Ligne constante')

# Ajouter des légendes et afficher le graphique
plt.legend()
plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.title('Tracer une ligne constante avec Matplotlib')
plt.show()