import numpy as np

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
    # print("A :",A,"l :",h,"E :",E,"G :",G, "Jx :",Jx, "Iy :",Iy, "Iz :",Iz) 
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
    return M_el, K_el
def matrice_locel(elements,dof_list) :
    """
    Crée la matrice locel
        Arguments :
            elements : liste des elements
            dof_list : liste des degrée de liberté
        Return :
            locel : matrice locel
    """
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
    return locel
def matrice_dof_list(nodes) :
    """
    Crée la matrice de degrée de liberté
        Arguments : 
            nodes : liste des noeuds
        return : 
            dof_list : liste des degrée de liberté
    """
    count = 1
    dof_list = []
    for e in range(len(nodes)) : 
        dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
        dof_list.append(dof_elem)
        count+=6

    return dof_list
def matrice_rotasion(nodes,elements,e) :
    """
    Crée la matrice de rotation
        Arguments : 
            nodes : liste des noeuds
            elements : liste des elements
            e : element courant
        return : 
            Te : matrice de rotation
    """
    node_1 = nodes[elements[e][0]] # mm
    node_2 = nodes[elements[e][1]]
    node_3 = [-1000, -1000, -1000] # pas colineaire
    d_2 = np.array([node_2[0]-node_1[0], node_2[1]-node_1[1], node_2[2]-node_1[2]])
    d_3 = np.array([node_3[0]-node_1[0], node_3[1]-node_1[1], node_3[2]-node_1[2]])
    elem_len = np.sqrt((node_2[0] - node_1[0])**2 + (node_2[1] - node_1[1])**2 + (node_2[2] - node_1[2])**2)
    dir_x    = d_2/elem_len
    dir_y    = np.cross(d_3, d_2)
    dir_y    = dir_y/np.linalg.norm(dir_y)
    dir_z    = np.cross(dir_x, dir_y)
    dir_X    = np.array([1.0, 0.0, 0.0])
    dir_Y    = np.array([0.0, 1.0, 0.0])
    dir_Z    = np.array([0.0, 0.0, 1.0])
    Re       = np.array([[np.dot(dir_X, dir_x), np.dot(dir_Y, dir_x), np.dot(dir_Z, dir_x)],[np.dot(dir_X, dir_y), np.dot(dir_Y, dir_y), np.dot(dir_Z, dir_y)],[np.dot(dir_X, dir_z), np.dot(dir_Y, dir_z), np.dot(dir_Z, dir_z)]])
    Te       = np.kron(np.eye(4), Re)
    return Te
def matrice_global(locel_e, K_eS, M_eS,dof_list) :
    """
    Crée la matrice globale
        Arguments : 
            locel_e : matrice locel
            K_eS : matrice de raideur
            M_eS : matrice de masse
            dof_list : liste des degrée de liberté
        return : 
            K : matrice de raideur globale
            M : matrice de masse globale
    """ 
    size = dof_list[len(dof_list)-1][5]
    K = np.zeros((size, size))
    M = np.zeros((size, size))
    locel_loc = locel_e
    for i in range(12) : 
        for j in range(12) : 
            ii = locel_loc[i]-1
            jj = locel_loc[j]-1 
            K[ii][jj] += K_eS[i][j]
            M[ii][jj] += M_eS[i][j]
    return K, M
