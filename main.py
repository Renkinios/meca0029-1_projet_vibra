import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import graphe as graphe
import math
from scipy import linalg
from scipy.linalg import eigh

# VISUALISATION DE LA STRUCTURE INITIALE
# si veux actualisé les graphes mettre true 
actu_graph = False 
nodes, elements = fct.read_data('Data/init_nodes.txt')
if actu_graph :
    graphe.plot_wind_turbine(nodes, elements)
    graphe.plot_rigid_links(nodes, elements)
    graphe.plot_nodes(nodes, elements, "picture/node_turbine_1.pdf")
nMode = 8 # nombre de mode a calculer,nombre de mode inclus dans al superoposition modale
# CREATION DES LISTES INITIALES DE CATEGORIE (leg ou rigid link (rili))
leg_elem = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
rili_elem = [56,57,58,59,60]
# MODIFICATION DU NOMBRE D'ELEMENTS
# elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements, leg_elem, rili_elem)
# fct.writing_nodes_element_file(nodes, elements, 'Data/nodes_2.txt')
# nodes, elements = fct.read_data('Data/nodes_2.txt')
# fct.plot_nodes(nodes, elements)

# CREATION DE LA LISTE DES DEGRES DE LIBERTE
count = 1
dof_list = []
for e in range(len(nodes)) : 
    dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
    dof_list.append(dof_elem)
    count+=6
#prend pas en compte les contrainte 
# CREATION DE LA MATRICE LOCEL (reprenant les dof impliques pour chaque element) 
locel = np.zeros((len(elements), 12))
for i in range(len(elements)) :   
    # locel[i][:] = [dof_list[elements[i][0]], dof_list[elements[i][1]]]
    locel[i][:] = [dof_list[elements[i][0]][0], dof_list[elements[i][0]][1], 
                   dof_list[elements[i][0]][2], dof_list[elements[i][0]][3], 
                   dof_list[elements[i][0]][4], dof_list[elements[i][0]][5],
                   dof_list[elements[i][1]][0], dof_list[elements[i][1]][1], 
                   dof_list[elements[i][1]][2], dof_list[elements[i][1]][3], 
                   dof_list[elements[i][1]][4], dof_list[elements[i][1]][5]]
locel = locel.astype(int)
# CREATION DES MATRICES ELEMENTAIRES, ROTATION ET ASSEMBLAGE
size = dof_list[len(dof_list)-1][5]
K = np.zeros((size, size))
M = np.zeros((size, size))
masse_total = 0 
# Boucle sur tous les elements
for e in range(len(elements)) : 
    # Creation des matries elementaires

    param = fct.get_param(e, leg_elem, rili_elem, elements, nodes)
    M_el, K_el = fct.elem_matrix(param)

    # Creation de l'operateur de rotation
    node_1 = nodes[elements[e][0]] # mm
    node_2 = nodes[elements[e][1]]
    node_3 = [-1000.0, 0.0, -1000.0] # pas colineaire

    d_2 = [node_2[0]-node_1[0], node_2[1]-node_1[1], node_2[2]-node_1[2]]
    d_3 = [node_3[0]-node_1[0], node_3[1]-node_1[1], node_3[2]-node_1[2]]

    elem_len = math.sqrt((node_2[0] - node_1[0])**2 + (node_2[1] - node_1[1])**2 + (node_2[2] - node_1[2])**2)
    dir_x = [(node_2[0]-node_1[0])/elem_len, (node_2[1]-node_1[1])/elem_len, (node_2[2]-node_1[2])/elem_len]
    dir_y = np.asarray(np.cross(d_2, d_3))/np.linalg.norm(np.cross(d_2, d_3))
    dir_z = np.cross(dir_x, dir_y)
    dir_X = [1.0, 0.0, 0.0]
    dir_Y = [0.0, 1.0, 0.0]
    dir_Z = [0.0, 0.0, 1.0]
    Re    = [[np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x)],[np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y)],[np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z)]]
    Te    = np.kron(np.eye(4), Re)
    K_eS  = Te.T@K_el@Te
    M_eS = Te.T@M_el@Te
    # Assemblage dans la matrice globale 
    locel_loc = locel[e]
    for i in range(12) : 
        for j in range(12) : 
            ii = locel_loc[i]-1
            jj = locel_loc[j]-1 # ? 
            K[ii][jj] += K_eS[i][j]
            M[ii][jj] += M_eS[i][j]
# AJOUT DE LA MASSE PONCTUELLE 
mass = np.diag([200000, 200000, 200000, 24e6, 24e6, 24e6]) #sur le dernier noeud de la jambe
dof_rotor = dof_list[21]
for m in range(6) : 
    for n in range(6) : 
        mm = dof_rotor[m]-1
        nn = dof_rotor[n]-1
        M[mm][nn] += mass[m][n]
#mode rigide masse, translation rotasion, terre , length 6 mode rigif 2.94 *10^57
# 
# print(np.sum(M - M.T))
# print(np.sum(K - K.T))
# APPLICATION DES CONTRAINTES 
for d in range(24) : 
    M = np.delete(M, (23-d), axis=0)
    M = np.delete(M, (23-d), axis=1)
    K = np.delete(K, (23-d), axis=0)
    K = np.delete(K, (23-d), axis=1)
u = np.zeros(len(M))
for i in range(0, len(M),6):
    u[i] = 1
masse_total += u.T@M@u
# numerical solution of K q= w^2 M q  juste K/M = w^2
# page 351 juste selectionner les n premiers modes 
# deja mis mm et EA/l ? 
print("Masse totale (kg) :", masse_total)
eigenvals, x = linalg.eigh(K,M)

sorted_indices = np.argsort(eigenvals)
eigenvals      = eigenvals[sorted_indices]
eigenvals      = eigenvals[-nMode:]
x              = x[sorted_indices]
x              = x[-nMode:]
w              = np.sqrt(eigenvals)
f              = w/(2*math.pi)
# D, V = eigh(K, M, 8) 
print("Fréquences propres (hz) :", f) 
# remet les valeur a 0 pour eigenvals
# graphe des modes de vibration deformation on rajoute les contraintes
if actu_graph :
    graphe.deformotion_frequence_propre(x,2,nodes,elements)
# ----------------------------------------------------- deuxieme partie --------------------------------------------------------------------
# Synchronous excitation in the form of a sine wave F = sin(wt) 
# sur le noeud 17 
#premier question crée la matrice C = aK + bM
#                                 𝜀_r = 1/2 (a w_0r + B/w_or)
# he damping ratio of the first two modes is equal to 0.5 % 
eps_1_2 = 0.005               
a       = 2*eps_1_2/(w[0] + w[1])
b       = a* w[0] * w[1]
C       = a * K + b * M
eps     = 0.5*(a*w+b/w)
print("Damping ratio :", eps)
exit_masse         = 1000              #kg
exit_vitesse       = 25 /3.6           #m/s
exit_frequence     = 1                 #hz
exit_temps_impacte = 0.05              #s
t                  = np.linspace(0,120,1000) 
F_max              = exit_masse*exit_vitesse*0.15 /exit_temps_impacte # delta momentum / delta t 
norm_F             = F_max *np.sin(2*np.pi*exit_frequence*t)            # regarder pour determiné comment appliquer la force
#force distrubué celon X et Y avec un angle de 45°
p = np.zeros((len(M),len(t)))
p[dof_list[17][0]-1] = norm_F/np.sqrt(2)
p[dof_list[17][1]-1] = norm_F/np.sqrt(2)

# Compute an approximate solution using the mode displacement method. Plot the
# time evolution in the direction of the impact both at the excitation point and
# at the rotor location.
# modale superposition method

q_deplacement, q_acc = fct.methode_superposition(M,K,w,x,eps,p,t,nMode)
# Compute the solution by time integration using the Newmark algorithm. Justify
# the choice of the time step and integration parameters. Plot the time evolution in
# the direction of the impact and its corresponding FFT, both at the excitation
# point and at the rotor location. regarder page 524-525 livres de references
q           = np.zeros((len(t),len(M)))
q_dot       = np.zeros((len(t),len(M)))
q_2dot      = np.zeros((len(t),len(M)))
q_star      = np.zeros((len(t),len(M)))
q_star_dot  = np.zeros((len(t),len(M)))

#chois gamma et beta corresponds to considering the acceleration average value
# : q_2dot(tau) = (q_2dot_n + q_2dot_n+1)/2
gamma = 1/2 
beta  = 1/4
h     = t[1]-t[0] #pas de temps
# condition initial  q_2dot = 0 & q_dot(0) = 0 & q(0) = 0  at the beginning no force
q[0]          = np.zeros(len(M))
q_dot[0]      = np.zeros(len(M))
q_2dot[0]     = np.zeros(len(M))
q_star[0]     = np.zeros(len(M))
q_star_dot[0] = np.zeros(len(M))
S             = M + gamma * h * C + beta * h**2 * K #retire de l'iteration pour l'optimsation
S_inverse     = np.linalg.inv(S)                    #retire de l'iteration pour l'optimsation
for n in range(len(t)-1) : 
    q_star_dot[n+1] = q_dot[n] + (1-gamma) * h * q_2dot[n]
    q_star[n+1]     = q[n] + h * q_dot[n] + (1/2-beta)*h**2*q_2dot[n]
    q_2dot[n+1]     = S_inverse @ (p.T[n+1] - C @ q_star_dot[n+1] - K @ q_star[n+1])
    q_dot[n+1]      = q_star_dot[n+1] + gamma * h * q_2dot[n+1]
    q[n+1]          = q_star[n+1] + beta * h**2 * q_2dot[n+1]