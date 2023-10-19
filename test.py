import matplotlib.pyplot as plt
import numpy as np

# Données
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Créez un subplot
fig, ax = plt.subplots()

# Tracez la courbe
ax.plot(x, y, label='sin(x)')

# Personnalisez les limites des axes
x_min, x_max = 0, 2 * np.pi
y_min, y_max = -1, 1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Personnalisez les graduations sur l'axe des abscisses
x_ticks = [0, np.pi, 2 * np.pi]
x_tick_labels = ['0', 'π', '2π']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)

# Personnalisez les graduations sur l'axe des ordonnées
y_ticks = [-1, 0, 1]
ax.set_yticks(y_ticks)

# Ajoutez des étiquettes aux axes
ax.set_xlabel('Axe des abscisses')
ax.set_ylabel('Axe des ordonnées')

# Réglez la taille des axes x et y
# Vous pouvez ajuster les valeurs pour obtenir la taille souhaitée
ax.set_aspect((x_max - x_min) / 2 / (y_max - y_min))

# Affichez le graphique
plt.show()
