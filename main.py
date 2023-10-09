import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math

# Visualisation de la structure initiale
nodes, elements = fct.read_data('init_nodes.txt')
fct.plot_nodes(nodes, elements)
# print(fct.euclidian_distance(6, elements, nodes))

# Création des listes initiales de catégorie (leg ou rigid link (rili))
leg_elem = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
rili_elem = [56,57,58,59,60]

# Modification du nombre d'éléments
elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements, leg_elem, rili_elem)
fct.writing_nodes_element_file(nodes, elements, 'nodes_2.txt')
# nodes, elements = fct.read_data('nodes_2.txt')
fct.plot_nodes(nodes, elements)
print(rili_elem)


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
    locel.append([dof_list[elements[i][0]][0], dof_list[elements[i][0]][1], 
                  dof_list[elements[i][0]][2], dof_list[elements[i][0]][3], 
                  dof_list[elements[i][0]][4], dof_list[elements[i][0]][5],
                  dof_list[elements[i][1]][0], dof_list[elements[i][1]][1], 
                  dof_list[elements[i][1]][2], dof_list[elements[i][1]][3], 
                  dof_list[elements[i][1]][4], dof_list[elements[i][1]][5]])


# Création des matrices élémentaires, rotation et assemblage
#boucle sur tous les éléments
for i in range(len(elements)) : 
    #création des matries élémentaires
    param = fct.get_param(i, leg_elem, rili_elem, elements, nodes)
    M_el,K_el = fct.elem_matrix(param)
    #création de l'opérateur de rotation
    # application de la rotation
    # assemblage dans la matrice globale



