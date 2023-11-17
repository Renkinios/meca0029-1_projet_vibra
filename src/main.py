import numpy as np
import math
from scipy import linalg
import functions as fct
import graphe as graphe
import write_read as write_read
import methode as mth
import matrice as mtx
# VISUALISATION DE LA STRUCTURE INITIALE
# si veux actualisé les graphes mettre true 
write_e_n       = True       # if you want to write the new nodes and element in a file
actu_graph      = False      # if you want actualisée graph
nb_elem_by_leg  = 2          #number of element by leg
nMode           = 8          # nombre de mode a calculer,nombre de mode inclus dans la superoposition modale
nodes, elements = write_read.read_data('Data/init_nodes.txt')

if actu_graph :
    graphe.plot_wind_turbine(nodes, elements)
    graphe.plot_rigid_links(nodes, elements)
    graphe.plot_nodes(nodes, elements, "picture/node_turbine_2.pdf")
# MODIFICATION DU NOMBRE D'ELEMENTS, CREATION DES LISTES INITIALES DE CATEGORIE

nodes,elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements,nb_elem_by_leg) 
if write_e_n :
    write_read.writing_nodes_element_file(nodes, elements, 'Data/nodes_test.txt')

# CREATION DE LA LISTE DES DEGRES DE LIBERTE
dof_list = mtx.matrice_dof_list(nodes)
#prend pas en compte les contrainte 
# CREATION DE LA MATRICE LOCEL (reprenant les dof impliques pour chaque element) 

locel    = mtx.matrice_locel(elements, dof_list)

# CREATION DES MATRICES ELEMENTAIRES, ROTATION ET ASSEMBLAGE
size = dof_list[len(dof_list)-1][5]
K = np.zeros((size, size))
M = np.zeros((size, size))

# Boucle sur tous les elements
for e in range(len(elements)) : 
    # Creation des matries elementaires
    param = fct.get_param(e, leg_elem, rili_elem, elements, nodes)
    M_el, K_el = mtx.elem_matrix(param)
    # Creation de l'operateur de rotation
    Te       = mtx.matrice_rotasion(nodes, elements, e)
    # Application de la rotation
    K_eS     = Te.T @ K_el @ Te
    M_eS     = Te.T @ M_el @Te
    # Assemblage des matrices global
    K, M = mtx.matrix_global_assembly(locel[e], K_eS, M_eS, K, M)

# AJOUT DE LA MASSE PONCTUELLE 
M = mtx.masse_ponctuelle(M, dof_list)
#mode rigide masse, translation rotasion, terre , length 6 mode rigif 2.94 *10^57
# 
# print(np.sum(M - M.T))
# print(np.sum(K - K.T))
# APPLICATION DES CONTRAINTES 
masse_total = fct.masse_total(M)
M, K        = fct.apply_constraints(M, K)
np.set_printoptions(threshold=np.inf, precision=8, suppress=True)
# CALCUL DES FREQUENCES PROPRES ET DES VECTEURS PROPRES
w,x = fct.natural_frequency(M, K,nMode)
print("Fréquences propres (hz) :", w/(2*np.pi)) 
print("Masse total :", masse_total)
# remet les valeur a 0 pour eigenvals
# graphe des modes de vibration deformation on rajoute les contraintes
if actu_graph :
    graphe.deformotion_frequence_propre(x,8,nodes,elements)
# ----------------------------------------------------- deuxieme partie --------------------------------------------------------------------
# Synchronous excitation in the form of a sine wave F = sin(wt) 
# sur le noeud 17 
# premier question crée la matrice C = aK + bM
#                                 𝜀_r = 1/2 (a w_0r + B/w_or)
# he damping ratio of the first two modes is equal to 0.5 % 
# directiond de l'impacte 45° et dans la direction de l'impacte au dernier noeud 
eps, C = mtx.damping_ratio(w,K,M)
print("Damping ratio :", eps)
# Compute the force p(t) applied at the excitation point.
t      = np.linspace(0, 10, 3000)
p      = mtx.force_p(M,dof_list,t) 

# Compute an approximate solution using the mode displacement method. Plot the
# time evolution in the direction of the impact both at the excitation point and
# at the rotor location.
# modale superposition method

q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,nMode)
if actu_graph :
    graphe.plot_q_deplacement(q_deplacement, dof_list,t, "picture/q_deplacement.pdf")
    graphe.plot_q_deplacement(q_acc, dof_list,t, "picture/q_acc.pdf")
# Compute the solution by time integration using the 
# . Justify
# the choice of the time step and integration parameters. Plot the time evolution in
# the direction of the impact and its corresponding FFT, both at the excitation
# point and at the rotor location. regarder page 524-525 livres de references
# Newmarks method
q = mth.New_mth(t,M,C,K,p)
if actu_graph :
    graphe.plot_q_deplacement(q, dof_list,t, "picture/q_newmark.pdf")
# ----------------------------------------------------- troisieme partie --------------------------------------------------------------------
K_gi,M_gi = mth.guyan_irons(dof_list,K,M)
w_Guyan_Irons,X_Guyan_Irons   = fct.natural_frequency(M_gi, K_gi,nMode)
print("Fréquences propres de Guyan-Irons (hz) :", w_Guyan_Irons/(2*np.pi)) 

# print(sol.shape)
# R_gi = [np.eye(8), -np.linalg.inv(Krr) @ Kcr] 