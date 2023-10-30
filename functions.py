import numpy as np
import matplotlib.pyplot as plt
import math

def read_data (file_name) : 
    """ Lit le fichier "fichier" contenant la liste des coordonnees des noeuds, et les elements
        Arguments : 
            - fichier : nom du fichier texte
        Return : 
            - nodes : la liste des coordonnees des noeuds 
            - elements : la liste des elements
    """

    # Initialiser les listes
    nodes = []
    elements = []

    # Ouvrir le fichier en mode lecture
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Ignorer la premiere ligne (Number of nodes)
    lines = lines[1:]
    cumpt = 0 

    # Parcourir les lignes restantes et inserer les coordonnees dans la nodes
    for line in lines:
        # Separer la ligne en tokens en utilisant l'espace comme delimiteur
        if("Number of elements :\n" == line):
            break
        tokens = line.split()
        cumpt += 1
        coordonnees = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        nodes.append(coordonnees)
    for line in range(cumpt+1, len(lines)):
        # Separer la ligne en tokens en utilisant l'espace comme delimiteur
        tokens = lines[line].split()
        elem_nodes = [int(tokens[2]), int(tokens[3])]
        elements.append(elem_nodes)

    return nodes, elements

def new_nodes(nodes, elements, leg_elem, rili_elem):
    """ Cree des nouveaux noeuds en divisant chaque elements en deux, et ajoute ces elements aux precedents
        Arguments : 
            - matrix : matrice contenant les noeuds initiaux    
        Return : 
            - new_matrix : matrice contenant les noeuds initiaux et les nouveaux noeuds
    """

    new_matrix = []
    new_leg = []
    new_rili = []
    count = 0
    for i in range(len(elements)):
        x = (nodes[elements[i][0]][0] + nodes[elements[i][1]][0])/2
        y = (nodes[elements[i][0]][1] + nodes[elements[i][1]][1])/2
        z = (nodes[elements[i][0]][2] + nodes[elements[i][1]][2])/2
        nodes.append([x, y, z])
        new_element_1 = [elements[i][0], len(nodes)-1]
        new_element_2 = [len(nodes)-1, elements[i][1]]
        new_matrix.append(new_element_1)
        new_matrix.append(new_element_2)
        if i in leg_elem : 
            new_leg.append(new_matrix.index(new_element_1))
            new_leg.append(new_matrix.index(new_element_2))
        if i in rili_elem : 
            new_rili.append(new_matrix.index(new_element_1))
            new_rili.append(new_matrix.index(new_element_2))
    return new_matrix, new_leg, new_rili

def writing_nodes_element_file(nodes,elements, file_name):
    """ Ecrit les nouveaux elements cres dans un fichier texte
        Arguments : 
            - nodes : liste des noeuds
            - elements : liste des elements 
        Return : 
            - Rien
    """

    with open(file_name, 'w') as fichier:
        fichier.write("Number of nodes " + str(len(nodes)) + " :\n")
        for i in range(len(nodes)):
            fichier.write("\t" + str(i) + " : " + str(nodes[i][0]) + " " + str(nodes[i][1]) + " " + str(nodes[i][2]) + "\n")
        fichier.write("Number of elements :\n")
        for i in range(len(elements)):
            fichier.write("\t"+ str(i) + " : " + str(elements[i][0]) + " " + str(elements[i][1]) + "\n")


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

def elem_matrix(beam_param) : 
    """ Cree les matrices elementaires
        Arguements : 
            - beam_param : une array contenant les parametres de la poutre : [A, r, h, E, Iz, Iy, Jx, G, rho]
        Return : 
            - M_el : la matrice elementaire de masse
            - K_el : la matrice elementaire d'energie cinetique
    """
    A = beam_param[0]; r = beam_param[1]; h=beam_param[2]; E=beam_param[3]; Iz=beam_param[4]; Iy=beam_param[5]; Jx=beam_param[6]; G=beam_param[7]; rho=beam_param[8]
    a = (11*h)/210
    b = (13*h)/420
    c = (h**2)/105
    d = (r**2)/3
    e = (h**2)/140
    M_el = [[  1/3,     0,     0,     0,     0,     0,   1/6,     0,     0,     0,     0,     0], 
            [    0, 13/35,     0,     0,     0,     a,     0,  9/70,     0,     0,     0,    -b],
            [    0,     0, 13/35,     0,    -a,     0,     0,     0,  9/70,     0,     b,     0], 
            [    0,     0,     0,     d,     0,     0,     0,     0,     0,   d/2,     0,     0],
            [    0,     0,    -a,     0,     c,     0,     0,     0,    -b,     0,    -e,     0], 
            [    0,     a,     0,     0,     0,     c,     0,     b,     0,     0,     0,    -e],
            [  1/6,     0,     0,     0,     0,     0,   1/3,     0,     0,     0,     0,     0],
            [    0,  9/70,     0,     0,     0,     b,     0, 13/35,     0,     0,     0,    -a],
            [    0,     0,  9/70,     0,    -b,     0,     0,     0, 13/35,     0,     a,     0],
            [    0,     0,     0,   d/2,     0,     0,     0,     0,     0,     d,     0,     0],
            [    0,     0,     b,     0,    -e,     0,     0,     0,     a,     0,     c,     0],
            [    0,    -b,     0,     0,     0,    -e,     0,    -a,     0,     0,     0,     c]]
    M_el = rho*A*h*np.array(M_el)
    f = (E*A)/h
    g = (12*E*Iz)/(h**3)
    i = (6*E*Iz)/(h**2) #g et j meme chose
    j = (12*E*Iy)/(h**3)
    k = (6*E*Iy)/(h**2)
    m = (G*Jx)/h
    n = (2*E*Iz)/h
    o = (2*E*Iy)/h
    K_el = [[    f,     0,     0,     0,     0,     0,    -f,     0,     0,     0,     0,     0], 
            [    0,     g,     0,     0,     0,     i,     0,    -g,     0,     0,     0,     i],
            [    0,     0,     j,     0,    -k,     0,     0,     0,    -j,     0,    -k,     0], 
            [    0,     0,     0,     m,     0,     0,     0,     0,     0,    -m,     0,     0],
            [    0,     0,    -k,     0,   2*o,     0,     0,     0,     k,     0,     o,     0], 
            [    0,     i,     0,     0,     0,   2*n,     0,    -i,     0,     0,     0,     n],
            [   -f,     0,     0,     0,     0,     0,     f,     0,     0,     0,     0,     0],
            [    0,    -g,     0,     0,     0,    -i,     0,     g,     0,     0,     0,    -i],
            [    0,     0,    -j,     0,     k,     0,     0,     0,     j,     0,     k,     0],
            [    0,     0,     0,    -m,     0,     0,     0,     0,     0,     m,     0,     0],
            [    0,     0,    -k,     0,     o,     0,     0,     0,     k,     0,   2*o,     0],
            [    0,     i,     0,     0,     0,     n,     0,    -i,     0,     0,     0,   2*n]]
    K_el = np.array(K_el)
    # print("matrice k_el", K_el)
    # print("matrice M_el : ",M_el)  
    return M_el, K_el
    
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
        m_leg = rho*math.pi*l*(0.5**2 - (0.5-0.02)**2)              # Masse d'une poutre principale [kg]
        Iyz_leg = (math.pi/64)*(1**4 - (1-0.04)**4)                  # Moment quadratique selon l'axe y et z [m^4]
        # Ix_leg = (math.pi/2)*(1**4 - (1-0.02)**2)                   # Moment quadratique selon l'axe x [m^4]
        Jx_leg = math.pi/4 * (0.5**4 - (0.5-0.02)**4)                    # Moment d'inertie selon l'axe x [kg.m^2]
        Jyz_leg = 0.25*m_leg*(1**2 + (1-0.02)**2)+(m_leg*(l**2))/12 # Moment d'inertie selon l'axe y et z [kg.m^2]
        r = np.sqrt(Jx_leg/A_leg)
        param = [A_leg, r, l, E, Iyz_leg, Iyz_leg, Jx_leg, G, rho]
        # print("leg_elem")
        # print("Aire :",param[0],"rayon :",param[1],"longeur element :",param[2],"Module de Young:",param[3],"Moment quadratique :",param[4],"Moment quadratique :",param[5],"moment d'inertie :",param[6])
    if elem in rili_elem : 
      # Definition des constantes pour les rigid links
        m_leg = rho*math.pi*l*(0.5**2 - (0.5-0.02)**2)  
        rho_r = rho*1e-4                                # Densite [kg/m^3]
        E_r = E*1e4                                     # Module de Young [Pa]
        A_r = A_leg*1e-2                                # Section [m^2]
        m_leg = rho_r*A_r*l                             # Masse [kg]
        Iyz_r = ((math.pi/64)*(1**4 - (1-0.04)**4))*1e4 # Moment quadratique selon l'axe y et z [m^4]
        # Jx_r = (0.5*m_leg*(0.5**2 + (0.5-0.02)**2))*1e4 # Moment d'intertie selon l'axe x [kg.m^2]
        Jx_r = math.pi/4 *(0.5**4 - (0.5-0.02)**4) *1e4      # Moment d'intertie selon l'axe x [m^4]
        G_r = E_r/(2*(1+nu))                            # Module de cisaillement [Pa]
        r = np.sqrt(Jx_r/A_r)
        param = [A_r, r, l, E_r, Iyz_r, Iyz_r, Jx_r, G_r, rho_r]
        # print("rigid elment")
        # print("Aire :",param[0],"rayon :",param[1],"longeur element :",param[2],"Module de Young:",param[3],"Moment quadratique :",param[4],"Moment quadratique :",param[5],"moment d'inertie :",param[6])
    else : 
      # Definition des caracteristiques pour une poutre secondaire (selon les axes locaux)
        m_beam = rho*math.pi*l*(0.3**2 - (0.3-0.02)**2)                    # Masse [kg]
        Iyz_beam = (math.pi/64)*(0.6**4 - (0.6-0.04)**4)                    # Moment quadratique selon l'axe y et z [m^4]
        Ix_beam = (math.pi/2)*(0.3**4 - (0.3-0.02)**2)                     # Moment quadratique selon l'axe x [m^4]
        # Jx_beam = 0.5*m_beam*(0.3**2 + (0.3-0.02)**2)                      # Moment d'inertie selon l'axe x [kg.m^2] 
        Jx_beam = math.pi/4 *(0.3**4 - (0.3-0.02)**4)
        Jyz_beam = 0.25*m_beam*(0.6**2 + (0.6-0.04)**2)+(m_beam*(l**2))/12 # Moment d'inertie selon l'axe y et z [km.m^2]
        r = np.sqrt(Jx_beam/A_beam)
        param = [A_beam,r, l, E, Iyz_beam, Iyz_beam, Jx_beam, G, rho]
        # print("little_element")
        # print("Aire :",param[0],"rayon :",param[1],"longeur element :",param[2],"Module de Young:",param[3],"Moment quadratique :",param[4],"Moment quadratique :",param[5],"moment d'inertie :",param[6])
    return param
