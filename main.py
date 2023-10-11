import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math

# Visualisation de la structure initiale
nodes, elements = fct.read_data('init_nodes.txt')
#fct.plot_nodes(nodes, elements)
# print(fct.euclidian_distance(6, elements, nodes))

# Création des listes initiales de catégorie (leg ou rigid link (rili))
leg_elem = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
rili_elem = [56,57,58,59,60]

# Modification du nombre d'éléments
elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements, leg_elem, rili_elem)
fct.writing_nodes_element_file(nodes, elements, 'nodes_2.txt')
# nodes, elements = fct.read_data('nodes_2.txt')
#fct.plot_nodes(nodes, elements)
#print(rili_elem)


# Création de la liste des degrés de liberté
count = 1
dof_list = []
for e in range(len(nodes)) : 
    dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
    dof_list.append(dof_elem)
    count+=6


# Création de la matrice locel (reprenant les dof impliqués pour chaque élément)
locel = []
for e in range(len(elements)) :    
    locel.append([dof_list[elements[e][0]][0], dof_list[elements[e][0]][1], 
                  dof_list[elements[e][0]][2], dof_list[elements[e][0]][3], 
                  dof_list[elements[e][0]][4], dof_list[elements[e][0]][5],
                  dof_list[elements[e][1]][0], dof_list[elements[e][1]][1], 
                  dof_list[elements[e][1]][2], dof_list[elements[e][1]][3], 
                  dof_list[elements[e][1]][4], dof_list[elements[e][1]][5]])



# Création des matrices élémentaires, rotation et assemblage
size = dof_list[len(dof_list)-1][5]
print(size)
K = np.zeros((size, size))
M = np.zeros((size, size))

# boucle sur tous les éléments
for e in range(len(elements)) : 
    # création des matries élémentaires
    param = fct.get_param(e, leg_elem, rili_elem, elements, nodes)
    M_el, K_el = fct.elem_matrix(param)

    #création de l'opérateur de rotation
    node_1 = nodes[elements[e][0]]
    node_2 = nodes[elements[e][1]]
    node_3 = [5000.0, 5000.0, 5000.0]

    d_2 = [node_2[0]-node_1[0], node_2[1]-node_1[1], node_2[2]-node_1[2]]
    d_3 = [node_3[0]-node_1[0], node_3[1]-node_1[1], node_3[2]-node_1[2]]

    elem_len = fct.euclidian_distance(e, elements, nodes)
    dir_x = [(node_2[0]-node_1[0])/elem_len, (node_2[1]-node_1[1])/elem_len, (node_2[2]-node_1[2])/elem_len]
    dir_y = np.cross(d_2, d_3)
    norme = np.linalg.norm(np.cross(d_2, d_3))
    for y in range(3) : 
        dir_y[y] = dir_y[y]/norme
    dir_z = np.cross(dir_x, dir_y)

    dir_X = [1.0, 0.0, 0.0]
    dir_Y = [0.0, 1.0, 0.0]
    dir_Z = [0.0, 0.0, 1.0]

    # print("\nProduit scalaire entre x et y :",np.dot(dir_x, dir_y))
    # print("\nProduit scalaire entre y et z :",np.dot(dir_y, dir_z))
    # print("\nProduit scalaire entre x et z :",np.dot(dir_x, dir_z))
    # print("\n la norme du vecteur x est de :",np.linalg.norm(dir_x))
    # print("\n la norme du vecteur y est de :",np.linalg.norm(dir_y))
    # print("\n la norme du vecteur z est de :",np.linalg.norm(dir_z))
    
    T = [[np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x), 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y), 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z), 0.0, 0.0, 0.0], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x)], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y)], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z)]]
    
    # application de la rotation
    k_eS = np.matmul(np.transpose(T), K_el)
    m_eS = np.matmul(np.transpose(T), M_el)
    K_eS = np.matmul(k_eS, T)
    M_eS = np.matmul(m_eS, T)

    # assemblage dans la matrice globale 
    locel_loc = locel[e]
    # print(locel_loc)
    # print(len(M))
    for i in range(12) : 
        for j in range(12) : 
            ii = locel_loc[i]-1
            jj = locel_loc[j]-1
            # print(ii)
            # print(jj)
            K[ii][jj] += K_eS[i][j]
            M[ii][jj] += M_eS[i][j]
    # # K[locel[e:]][locel[e:]] = K[locel[e:]][locel[e:]] + K_eS
    # # M[locel[e:]][locel[e:]] = M[locel[e:]][locel[e:]] + M_eS
print(K)