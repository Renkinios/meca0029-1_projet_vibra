import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math
from scipy import linalg
from scipy.linalg import eigh

# VISUALISATION DE LA STRUCTURE INITIALE
nodes, elements = fct.read_data('init_nodes.txt')
# fct.plot_nodes(nodes, elements)

# CREATION DES LISTES INITIALES DE CATEGORIE (leg ou rigid link (rili))
leg_elem = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
rili_elem = [56,57,58,59,60]

# MODIFICATION DU NOMBRE D'ELEMENTS
# elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements, leg_elem, rili_elem)
# fct.writing_nodes_element_file(nodes, elements, 'nodes_2.txt')
# nodes, elements = fct.read_data('nodes_2.txt')
# fct.plot_nodes(nodes, elements)

# CREATION DE LA LISTE DES DEGRES DE LIBERTE
count = 1
dof_list = []
for e in range(len(nodes)) : 
    dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
    dof_list.append(dof_elem)
    count+=6

# CREATION DE LA MATRICE LOCEL (reprenant les dof impliques pour chaque element) 
locel = []
for e in range(len(elements)) :    
    locel.append([dof_list[elements[e][0]][0], dof_list[elements[e][0]][1], 
                  dof_list[elements[e][0]][2], dof_list[elements[e][0]][3], 
                  dof_list[elements[e][0]][4], dof_list[elements[e][0]][5],
                  dof_list[elements[e][1]][0], dof_list[elements[e][1]][1], 
                  dof_list[elements[e][1]][2], dof_list[elements[e][1]][3], 
                  dof_list[elements[e][1]][4], dof_list[elements[e][1]][5]])



# CREATION DES MATRICES ELEMENTAIRES, ROTATION ET ASSEMBLAGE
size = dof_list[len(dof_list)-1][5]
K = np.zeros((size, size))
M = np.zeros((size, size))

# Boucle sur tous les elements
for e in range(len(elements)) : 
    # Creation des matries elementaires
    param = fct.get_param(e, leg_elem, rili_elem, elements, nodes)
    M_el, K_el = fct.elem_matrix(param)

    # Creation de l'operateur de rotation
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
    
    # Application de la rotation
    k_eS = np.matmul(np.transpose(T), K_el)
    m_eS = np.matmul(np.transpose(T), M_el)
    K_eS = np.matmul(k_eS, T)
    M_eS = np.matmul(m_eS, T)

    # Assemblage dans la matrice globale 
    print(locel[0:])
    locel_loc = locel[e]
    for i in range(12) : 
        locel_loc[i] -= 1
    K[locel_loc][locel_loc] += K_eS
    M[locel_loc][locel_loc] += M_eS
    # locel_loc = locel[e]
    # for i in range(12) : 
    #     for j in range(12) : 
    #         ii = locel_loc[i]-1
    #         jj = locel_loc[j]-1 # ? 
    #         K[ii][jj] += K_eS[i][j]
    #         M[ii][jj] += M_eS[i][j]

# TEST DE SYMETRIE DES MATRICES 
if(np.array_equal(M, np.transpose(M)) == True) : 
    print("Matrice M : OK")
if(np.array_equal(M, np.transpose(M)) == False) : 
    print("Matrice M : PAS OK")

if(np.array_equal(K, np.transpose(K)) == True) : 
    print("Matrice K : OK")
if(np.array_equal(K, np.transpose(K)) == False) : 
    print("Matrice K : PAS OK")

# AJOUT DE LA MASSE PONCTUELLE 
mass = np.diag([200000, 200000, 200000, 24e6, 24e6, 24e6])
dof_rotor = dof_list[21]
for m in range(6) : 
    for n in range(6) : 
        mm = dof_rotor[m]-1
        nn = dof_rotor[n]-1
        M[mm][nn] += mass[m][n]


# APPLICATION DES CONTRAINTES 
for d in range(24) : 
    M = np.delete(M, (23-d), axis=0)
    M = np.delete(M, (23-d), axis=1)
    K = np.delete(K, (23-d), axis=0)
    K = np.delete(K, (23-d), axis=1)

# numerical solution of K q= w^2 M q  juste K/M = w^2
# page 351 juste selectionner les n premiers modes 
#deja mis mm et EA/l ? 

eigenvals, eigenvects = linalg.eigh(K,M)
eigenvals = np.sort(eigenvals)

eigenvals = eigenvals[-8:]
w = np.sqrt(eigenvals)
f = w/(2*math.pi)
# D, V = eigh(K, M, 8) 
# print(f)
# print("Frequences propres (rad/s) :", D)  
