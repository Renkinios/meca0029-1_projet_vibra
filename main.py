import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math

# Visualisation de la structure initiale
nodes, elements = fct.read_data('init_nodes.txt')
# fct.plot_nodes(nodes, elements)
# print(fct.euclidian_distance(6, elements, nodes))

# Modification du nombre d'éléments
# elements = fct.new_nodes(nodes, elements)
# fct.writing_nodes_element_file(nodes, elements, 'nodes_2.txt')
# nodes, elements = fct.read_data('nodes_2.txt')
# fct.plot_nodes(nodes, elements)

# Création de la liste des degrés de liberté
count = 1
dof_list = []
for i in range(len(nodes)) : 
    dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
    dof_list.append(dof_elem)
    count+=6


# Création de la matrice locel (reprenant les dof impliqués pour chaque élément)
locel = []
for i in range(len(elements)) :    
    locel.append([dof_list[elements[i][0]][0],dof_list[elements[i][0]][1], dof_list[elements[i][0]][2], dof_list[elements[i][0]][3], dof_list[elements[i][0]][4], dof_list[elements[i][0]][5],
                  dof_list[elements[i][1]][0], dof_list[elements[i][1]][1], dof_list[elements[i][1]][2], dof_list[elements[i][1]][3], dof_list[elements[i][1]][4], dof_list[elements[i][1]][5]])


# Définition des constantes 
E = 210e6                             # Module de Young [Pa]
A_leg = math.pi*(1 - 0.02**2)         # Section d'une poutre principale [m^2]
A_beam = math.pi*(0.6**2 - 0.02**2)   # Section d'une poutre secondaire [m¨2]


