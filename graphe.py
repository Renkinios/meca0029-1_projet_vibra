import matplotlib.pyplot as plt
import functions  as fct
import numpy as np
nodes, elements = fct.read_data('Data/init_nodes.txt')
nodes_2, elements_2 = fct.read_data('Data/nodes_2.txt')
def plot_wind_turbine(nodes, elements) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,20,21,22,23,32,33,34,35]
    fig = plt.figure(figsize=((20,100)))
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in main_legs : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="red")
        elif j in [56,57,58,59,60] : 

            pass
        else :
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue")
        j += 1
    for i in range(4) : 
        ax.scatter(nodes[i][0]/1000, nodes[i][1]/1000, nodes[i][2]/1000,color='orange', marker='o')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]', labelpad=100)
    ax.set_aspect('equal')
    plt.savefig('picture/wind_turbine.pdf',bbox_inches='tight',dpi=600,format='pdf')
plot_wind_turbine(nodes, elements) 

def plot_rigid_links(nodes, elements) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,20,21,22,23,32,33,34,35]
    fig = plt.figure(figsize=((20,100)))
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in main_legs : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="red")
        elif j in [56,57,58,59,60] : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="green")
            ax.scatter(nodes[21][0]/1000, nodes[21][1]/1000, nodes[21][2]/1000,color='green', marker='o')
        else :
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue")
        j += 1
    for i in range(4) : 
        ax.scatter(nodes[i][0]/1000, nodes[i][1]/1000, nodes[i][2]/1000,color='orange', marker='o')

    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]', labelpad=100)
    ax.set_aspect('equal')
    plt.savefig('picture/rigid_links.pdf',bbox_inches='tight',dpi=600,format='pdf')
plot_rigid_links(nodes, elements)
def plot_nodes(nodes, elements,fichier) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,20,21,22,23,32,33,34,35]
    fig = plt.figure(figsize=((20,100)))
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in main_legs : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue")
        elif j in [56,57,58,59,60] : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="green")
            ax.scatter(nodes[21][0]/1000, nodes[21][1]/1000, nodes[21][2]/1000,color='green', marker='o')
        else :
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue")
        j += 1
    for node in nodes:
        ax.plot(node[0]/1000, node[1]/1000, node[2]/1000, 'peru', marker='o', markersize = 3)
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]', labelpad=100)
    ax.set_aspect('equal')
    plt.savefig(fichier,bbox_inches='tight',dpi=600,format='pdf')
plot_nodes(nodes_2, elements_2,"picture/node_turbine_2.pdf") 
plot_nodes(nodes, elements,"picture/node_turbine_1.pdf")
# convergence study
def convergence_study() : 
    number_element = []
    frequence = [[4.42401783E-01,4.49817787E-01,9.60608860E-01,6.89354897E+00,7.32969389E+00,1.63825977E+01,
    2.04311445E+01,2.22578560E+01],[4.42402461E-01,4.49817194E-01,9.60603890E-01,6.87080562E+00,7.30317814E+00,
    1.63451574E+01,1.97419262E+01,2.14418085E+01],[4.42402506E-01,4.49817158E-01,9.60603607E-01,6.86900918E+00,7.30109435E+00,1.63422428E+01,
    1.96804680E+01,2.13663317E+01],[4.42402506E-01,4.49817158E-01,9.60603607E-01,6.86900918E+00,7.30109435E+00,
    1.63422428E+01,1.96804680E+01,2.13663317E+01]]
    for i in range (1,len(frequence)+1) : 
        number_element.append(16*i + 28*i + 5)
    dif = [] #matrice qui va calculé le decalage par rapport au nombre d'élémént
    # total_diff = 0  
    # print(len(frequence)-1)
    # for i in range (len(frequence)) : 
    #     total_diff = 0
    #     for j in range(8) : 
    #         total_diff += abs(frequence[0][j] - frequence[i][j])
    #     dif.append(total_diff)
    f_propre_1 = []
    f_propre_2 = []
    f_propre_3 = []
    for i in range(len(frequence)): 
        f_propre_1.append(frequence[i][1])
        f_propre_2.append(frequence[i][3])
        f_propre_3.append(frequence[i][5])
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(number_element, f_propre_1, label = "2st frequency") 
    ax.plot(number_element, f_propre_2, label = "4nd frequency")
    ax.plot(number_element, f_propre_3, label = "6rd frequency")
    ax.axvline(x=93, color='red', linestyle='--', label='converege')
    ax.legend(loc="upper right")
    ax.set_xticks(number_element)
    ax.set_xlabel('Number elements[/]') 
    ax.set_ylabel('Frequency [Hz]')
    plt.savefig("picture/convergence_study.pdf", bbox_inches="tight", dpi=600)   
    
# convergence_study() 
def plot_q_deplacement(x,t) : 
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.plot(t, x) 
    ax.set_xlabel('Time [s]') 
    ax.set_ylabel('Displacement [m]')
    plt.savefig("picture/q_deplacement.pdf", bbox_inches="tight", dpi=600)
def deformotion_frequence_propre(x,nMode,nodes,elements) :
    titre = "picture/py_def_mode_"
    matrix_def = np.copy(x)
    for i in range(24) :
        matrix_def = np.insert(matrix_def,0,0,axis=1)
    for mode in range(nMode) :
        titre += str(mode+1) + ".pdf"
        fig = plt.figure(figsize=((20,100)))
        ax  = fig.add_subplot(111, projection='3d')
        for i in nodes : 
            ax.scatter(i[0]/1000, i[1]/1000, i[2]/1000,color='lightblue', marker='o',alpha=0.3)
        for j in elements : 
            x = [nodes[j[0]][0]/1000, nodes[j[1]][0]/1000]
            y = [nodes[j[0]][1]/1000, nodes[j[1]][1]/1000]
            z = [nodes[j[0]][2]/1000, nodes[j[1]][2]/1000]
            ax.plot(x, y, z,color="navy")
        defo_nodes = nodes.copy()
        defo_nodes = np.array(defo_nodes)/1000
        v = 0
        for i in range(len(defo_nodes)) :
            v = 6 * i
            defo_nodes[i][0] += 10  * matrix_def[mode][v]
            defo_nodes[i][1] += 10 * matrix_def[mode][v+1]
            defo_nodes[i][2] += 10 * matrix_def[mode][v+2]
        for i in elements : 
            x = [defo_nodes[i[0]][0], defo_nodes[i[1]][0]]
            y = [defo_nodes[i[0]][1], defo_nodes[i[1]][1]]
            z = [defo_nodes[i[0]][2], defo_nodes[i[1]][2]]
            ax.plot(x, y, z,color="red")
        ax.set_xlabel('X-axis [m]')
        ax.set_ylabel('Y-axis [m]')
        ax.set_zlabel('Z-axis [m]')
        ax.set_zlabel('Z-axis [m]', labelpad=100)
        ax.set_aspect('equal')
        plt.savefig(titre, bbox_inches="tight", dpi=600)