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
write_e_n       = False       # if you want to write the new nodes and element in a file
actu_graph      = False      # if you want actualisée graph
nb_elem_by_leg  = 4          #number of element by leg
nMode           = 8          # nombre de mode a calculer,nombre de mode inclus dans la superoposition modale
nodes, elements = write_read.read_data('Data/init_nodes.txt')

if actu_graph :
    graphe.plot_wind_turbine(nodes, elements)
    graphe.plot_rigid_links(nodes, elements)
    graphe.plot_nodes(nodes, elements, "picture/node_turbine_2.pdf")
# MODIFICATION DU NOMBRE D'ELEMENTS, CREATION DES LISTES INITIALES DE CATEGORIE

nodes,elements, leg_elem, rili_elem = fct.new_nodes(nodes, elements,nb_elem_by_leg) 
if write_e_n :
    write_read.writing_nodes_element_file(nodes, elements, 'Data/nodes_1.txt')

# CREATION DE LA LISTE DES DEGRES DE LIBERTE
dof_list = mtx.matrice_dof_list(nodes)
#prend pas en compte les contrainte 
# CREATION DE LA MATRICE LOCEL (reprenant les dof impliques pour chaque element) 

locel = mtx.matrice_locel(elements, dof_list)

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
    K_eS     = Te.T @ K_el @ Te
    M_eS     = Te.T @ M_el @Te

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
    mm         = dof_rotor[m]-1
    nn         = dof_rotor[m]-1
    M[mm][nn] += mass[m][m]
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
# if actu_graph :
graphe.deformotion_frequence_propre(x,8,nodes,elements)
# ----------------------------------------------------- deuxieme partie --------------------------------------------------------------------
# Synchronous excitation in the form of a sine wave F = sin(wt) 
# sur le noeud 17 
# premier question crée la matrice C = aK + bM
#                                 𝜀_r = 1/2 (a w_0r + B/w_or)
# he damping ratio of the first two modes is equal to 0.5 % 
# directiond de l'impacte 45° et dans la direction de l'impacte au dernier noeud 
eps_1_2 = 0.005               
a       = 2*eps_1_2/(w[0] + w[1])
b       = a* w[0] * w[1]
C       = a * K + b * M
eps     = 0.5*(a*w+b/w)
print("Damping ratio :", eps)
exit_masse           = 1000              #kg
exit_vitesse         = 25 /3.6           #m/s
exit_frequence       = 1                 #hz
exit_temps_impacte   = 0.05              #s
t                    = np.linspace(0,120,1000) 
F_max                = exit_masse*exit_vitesse*0.15 /exit_temps_impacte   #[N]
print("Force max :", F_max) # delta momentum / delta t  
norm_F               = F_max *np.sin(2*np.pi*exit_frequence*t)            # regarder pour determiné comment appliquer la force
#force distrubué celon X et Y avec un angle de 45°
p                    = np.zeros((len(M),len(t)))
p[dof_list[17][0]-1] = norm_F/np.sqrt(2)
p[dof_list[17][1]-1] = norm_F/np.sqrt(2)

# Compute an approximate solution using the mode displacement method. Plot the
# time evolution in the direction of the impact both at the excitation point and
# at the rotor location.
# modale superposition method

q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,nMode)
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