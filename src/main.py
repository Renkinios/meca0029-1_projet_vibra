import numpy as np
import math
from scipy import linalg
import functions as fct
import graphe as graphe
import write_read as write_read
import methode as mth
import matrice as mtx
import time
# VISUALISATION DE LA STRUCTURE INITIALE
# si veux actualisé les graphes mettre true 
write_e_n       = False       # if you want to write the new nodes and element in a file
actu_graph      = False      # if you want actualisée graph
nb_elem_by_leg  = 3          #number of element by leg
nMode           = 8          # nombre de mode a calculer,nombre de mode inclus dans la superoposition modale
nodes, elements = write_read.read_data('Data/init_nodes.txt')
#----------------------------------------------------- premiere partie --------------------------------------------------------------------
if actu_graph :
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

# APPLICATION DES CONTRAINTES 
masse_total = fct.masse_total(M)
M, K        = fct.apply_constraints(M, K)
np.set_printoptions(threshold=np.inf, precision=8, suppress=True)
# CALCUL DES FREQUENCES PROPRES ET DES VECTEURS PROPRES
w,x = fct.natural_frequency(M, K,nMode)
print("Fréquences propres (hz) :", w/(2*np.pi)) 
print("Masse total :", masse_total)
# graphe des modes de vibration deformation on rajoute les contraintes
if actu_graph :
    graphe.deformotion_frequence_propre(x,8,nodes,elements)
# ----------------------------------------------------- deuxieme partie --------------------------------------------------------------------
# Synchronous excitation in the form of a sine wave F = sin(wt) 
# directiond de l'impacte 45° et dans la direction de l'impacte au dernier noeud 
eps, C = mtx.damping_ratio(w,K,M)
print("Damping ratio :", eps)
# Compute the force p(t) applied at the excitation point.
t      = np.linspace(0, 10, 3000)
p      = mtx.force_p(M,dof_list,t) 

#q_deplacement, q_acceleration
# Newmark method
q = mth.New_mth(t,M,C,K,p)
if actu_graph :
    graphe.plot_q_deplacement(q, dof_list,t, "picture/q_newmark.pdf")
    graphe.conv_time_new(t,M,C,K,dof_list,True)    
    graphe.conv_time_new(t,M,C,K,dof_list,False)
graphe.fft_new_R(q,t,dof_list)


#q_deplacement, q_acceleration
q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,nMode)

if actu_graph :
    graphe.plot_q_deplacement(q_deplacement, dof_list,t, "picture/q_deplacement.pdf")
    graphe.plot_q_deplacement(q_acc, dof_list,t, "picture/q_acc.pdf")
    graphe.comp_depl_acc_newR(q_deplacement,q_acc,q,t,dof_list)
    graphe.conp_Mode_dep(M,K,w,x,eps,p,t,dof_list,nMode,True)
    graphe.conp_Mode_acc(M,K,w,x,eps,p,t,dof_list,nMode,True)
    graphe.conp_Mode_dep(M,K,w,x,eps,p,t,dof_list,nMode,False)
    graphe.conp_Mode_acc(M,K,w,x,eps,p,t,dof_list,nMode,False)

# ----------------------------------------------------- troisieme partie --------------------------------------------------------------------
# method de guyan irons
K_gi, M_gi, Mcc, Kcc, Rgi, Krr, Kt, Mt,dofR,Cdofs = mth.guyan_irons(dof_list,K,M,nMode)
w_gi, x_gi = fct.natural_frequency(M_gi, K_gi,nMode) 
print("Fréquences propres guyan_irons(hz) :", w_gi/(2*np.pi))

# method de Craig Bampton 
Neigenmodes      = 8
K_cb, M_cb, Rcb  = mth.Craig_Bampton(Mcc,Kcc,Krr,Rgi,Neigenmodes,nMode,Kt,Mt)
w_cb, x_cb       = fct.natural_frequency(M_cb, K_cb,nMode)
print("Fréquences propres Craig_Bampton(hz) :", w_cb/(2*np.pi))

#methode de Craig Bampton avec Newmark
Crr   = C[np.ix_(dofR, dofR)]     # retained part
Crc   = C[np.ix_(dofR, Cdofs)]
Ccc   = C[np.ix_(Cdofs, Cdofs)]   # condensed part
Ccr   = C[np.ix_(Cdofs, dofR)]
C_t   = np.block([[Crr, Crc], [Ccr, Ccc]])
C_new = Rcb.T @ C_t @ Rcb

p_r   = p[np.ix_(dofR)]
p_c   = p[np.ix_(Cdofs)]
p_t   = np.block([[p_r], [p_c]])
p_new = Rcb.T @ p_t

q_app = mth.New_mth(t,M_cb,C_new,K_cb,p_new)

if actu_graph :
    graphe.comp_newR_new_R_ap(q,q_app,dof_list,t)
    graphe.comp_Craig_guyan(Mcc,Kcc,Krr,Rgi,Kt,Mt,w_gi,8,nMode,w)
