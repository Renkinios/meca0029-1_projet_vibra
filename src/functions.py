import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg

def new_nodes(nodes, elements,nb_elemnt_by_leg):
    """ Cree des nouveaux noeuds en divisant chaque elements en deux, et ajoute ces elements aux precedents
        Arguments : 
            - matrix : matrice contenant les noeuds initiaux    
        Return : 
            - new_element : matrice contenant les noeuds initiaux et les nouveaux noeuds
    """
    leg_elem    = [0,1,2,3,8,9,10,11,20,21,22,23,32,33,34,35]
    rili_elem   = [44,45,46,47,48]
    new_nodes   = [row[:] for row in nodes]
    new_element = []
    new_leg     = []
    new_rili    = []
    count = len(nodes)
    for i in range(len(elements)):
        if nb_elemnt_by_leg == 0 :
            KeyError("nb_elemnt_by_leg doit etre superieur a 0")
        elif nb_elemnt_by_leg == 1 :
            if i in rili_elem :
                new_element_1 = [elements[i][0], elements[i][1]]
                new_element.append(new_element_1)
                new_rili.append(new_element.index(new_element_1))
            else :
                new_element_1 =  [elements[i][0], elements[i][1]]
                new_element.append(new_element_1)
        for j in range(1,nb_elemnt_by_leg) :
            if i in rili_elem :
                if j ==1 :
                    new_element_1 = [elements[i][0], elements[i][1]]
                    new_element.append(new_element_1)
                    new_rili.append(new_element.index(new_element_1))
                else :
                    pass 
            else :
                x = nodes[elements[i][0]][0] + j * (nodes[elements[i][1]][0] - nodes[elements[i][0]][0])/nb_elemnt_by_leg
                y = nodes[elements[i][0]][1] + j * (nodes[elements[i][1]][1] - nodes[elements[i][0]][1])/nb_elemnt_by_leg
                z = nodes[elements[i][0]][2] + j * (nodes[elements[i][1]][2] - nodes[elements[i][0]][2])/nb_elemnt_by_leg
                new_nodes.append([x, y, z])
                if j == 1 and j+1 == nb_elemnt_by_leg :
                    new_element_1 = [elements[i][0], count]
                    new_element.append(new_element_1)
                    new_element.append([count, elements[i][1]])
                elif j+1 == nb_elemnt_by_leg :
                    new_element.append([count-1, count])
                    new_element.append([count, elements[i][1]])
                elif j == 1 :
                    new_element_1 = [elements[i][0], count]
                    new_element.append(new_element_1)
                else :
                    new_element.append([count-1, count])
                count += 1
        if i in leg_elem :
            for j in range(nb_elemnt_by_leg) :   
                new_leg.append(new_element.index(new_element_1)+j)
    return new_nodes, new_element, new_leg, new_rili



def euclidian_distance(elem, elements, nodes) : 
    """ Calcule la longueur de l'element via la formule de la distance euclidienne
        Argument : 
            - elem : scalaire donnant l'index de l'element dans la liste elements
            - elements : la liste des elements
            - nodes : la liste des coordonnees de chaque noeud
        Return : 
            - elem_len : la longueur de l'element
    """

    node_1 = elements[elem][0]
    node_2 = elements[elem][1]
    coord_1 = nodes[node_1]
    coord_2 = nodes[node_2]

    elem_len = math.sqrt(((coord_1[0]-coord_2[0])**2)+((coord_1[1]-coord_2[1])**2)+((coord_1[2]-coord_2[2])**2))

    return elem_len

def get_param(elem, leg_elem, rili_elem, elements, nodes) : 
    """ Cree la liste des parametres en fonction de la categorie de l'element (leg, rigid link or beam)
        Arguments : 
            - elem : l'index de l'elements concerne
            - leg_elem : la liste d'index des elements appartenants aux legs
            - rili_elem : la liste d'index des elements appartenants aux rigid links
            - elements : la liste ldes elements
            - nodes : la liste des noeuds
        Return : 
            - param : la liste des parametres de l'element : [A, r, h, E, Iz, Iy, Jx, G, rho]
    """

    # Definition des constantes pour la structure
    E = 210e9                                           # Module de Young [Pa]
    A_leg = math.pi*(0.5**2 - (0.5-0.02)**2)            # Section d'une poutre principale [m^2]
    A_beam = math.pi*(0.3**2 - (0.3-0.02)**2)           # Section d'une poutre secondaire [m^2]
    nu = 0.3                                            # Coefficient de Poisson [-]
    rho = 7800                                          # Densite de l'acier utilise [kg/m^3]
    G = E/(2*(1+nu))                                    # Module de cisaillement [Pa]
    l = euclidian_distance(elem, elements,nodes)        
    l = l*1e-3                                          # Longueur de l'element [m]

    param = []

    if elem in leg_elem : 
      # Definition des caracteristiques pour une pour principale (elon les axes locaux)
        Iyz_leg = (math.pi / 64) * (1**4 - 0.96**4)  # Moment quadratique selon l'axe y et z [m^4]
        Jx_leg = Iyz_leg * 2                         # Moment quadratique selon l'axe x [m^4]
        r_leg = np.sqrt(Jx_leg/A_leg)               # Rayon de giration [m]
        param = [A_leg, r_leg, l, E, Iyz_leg, Iyz_leg, Jx_leg, G, rho]


    elif elem in rili_elem : 
      # Definition des constantes pour les rigid links
        rho_r = rho*1e-4                                # Densite [kg/m^3]
        E_r = E*1e4                                     # Module de Young [Pa]
        A_r = A_leg*1e-2                                # Section [m^2]
        Iyz_r = ((math.pi / 64) * (1**4 - 0.96**4))*1e4 # Moment quadratique selon l'axe y et z [m^4]
        Jx_r = (Iyz_r * 2)                              # Moment quadratique selon l'axe x [m^4]
        G_r = E_r/(2*(1+nu))                            # Module de cisaillement [Pa]
        r_rili = np.sqrt(Jx_r/A_r)                      # Rayon de giration [m]
        param = [A_r, r_rili, l, E_r, Iyz_r, Iyz_r, Jx_r, G_r, rho_r]
    

    else : 
      # Definition des caracteristiques pour une poutre secondaire (selon les axes locaux)
        Iyz_beam = (math.pi / 64) * (0.6**4 - 0.56**4)  # Moment quadratique selon l'axe y et z [m^4]
        Jx_beam = Iyz_beam * 2                          # Moment quadratique selon l'axe x [m^4]
        r_beam = np.sqrt(Jx_beam/A_beam)               # Rayon de giration [m]
        param = [A_beam,r_beam, l, E, Iyz_beam, Iyz_beam, Jx_beam, G, rho]
    
    return param

def apply_constraints(M,K) :
    """
    Applique les contraintes
        Arguments : 
            M : matrice de masse
            K : matrice de raideur
        return : 
            M : matrice de masse avec les contraintes
            K : matrice de raideur avec les contraintes
    """
    for d in range(24) : 
        M = np.delete(M, (23-d), axis=0)
        M = np.delete(M, (23-d), axis=1)
        K = np.delete(K, (23-d), axis=0)
        K = np.delete(K, (23-d), axis=1)
    return M, K

def masse_total(M) : 
    """
    Calcule la masse totale
        Arguments : 
            M : matrice de masse
        return : 
            masse_total : masse totale
    """
    u = np.zeros(len(M))
    for i in range(0, len(M),6):
        u[i] = 1
    masse_total = u.T@M@u
    return masse_total
def natural_frequency(M,K,nMode) :
    """
    Calcule les fréquences naturelles
        Arguments : 
            M : matrice de masse
            K : matrice de raideur
            nMode : nombre de mode repris
        return : 
            w : fréquences naturelles
            x : vecteur propre
    """
    eigenvals, x   = linalg.eig(K,M)
    sorted_indices = np.argsort(eigenvals)
    eigenvals      = eigenvals[sorted_indices]
    eigenvals      = eigenvals[:nMode]
    x              = x[:, sorted_indices]
    w              = np.real(np.sqrt(eigenvals))

    return w, x