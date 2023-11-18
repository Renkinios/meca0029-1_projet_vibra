import matplotlib.pyplot as plt
import functions  as fct
import numpy as np
import write_read as write_read

def plot_nodes(nodes, elements,fichier) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    nodes, elements = write_read.read_data('Data/init_nodes.txt')
    nodes_2, elements_2,leg_elem_2,rili_elem_2 = fct.new_nodes(nodes, elements,4)
    node_4, element_4,leg_elem_4,rili_elem_4 = fct.new_nodes(nodes_2, elements_2,2)
    fig = plt.figure(figsize=((40,5)))
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in leg_elem_2 : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="red")
        elif j in rili_elem_2: 
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
# plot_nodes(nodes_2, elements_2,"picture/node_turbine_2.pdf") 
# plot_nodes(nodes, elements,"picture/node_turbine_1.pdf")
# convergence study
# def convergence_study() : 
#     number_element = []
#     frequence = [[4.42401783E-01,4.49817787E-01,9.60608860E-01,6.89354897E+00,7.32969389E+00,1.63825977E+01,
#     2.04311445E+01,2.22578560E+01],[4.42402461E-01,4.49817194E-01,9.60603890E-01,6.87080562E+00,7.30317814E+00,
#     1.63451574E+01,1.97419262E+01,2.14418085E+01],[4.42402506E-01,4.49817158E-01,9.60603607E-01,6.86900918E+00,7.30109435E+00,1.63422428E+01,
#     1.96804680E+01,2.13663317E+01],[4.42402506E-01,4.49817158E-01,9.60603607E-01,6.86900918E+00,7.30109435E+00,
#     1.63422428E+01,1.96804680E+01,2.13663317E+01]]
#     for i in range (1,len(frequence)+1) : 
#         number_element.append(16*i + 28*i + 5)
#     dif = [] #matrice qui va calculé le decalage par rapport au nombre d'élémént
#     # total_diff = 0  
#     # print(len(frequence)-1)
#     # for i in range (len(frequence)) : 
#     #     total_diff = 0
#     #     for j in range(8) : 
#     #         total_diff += abs(frequence[0][j] - frequence[i][j])
#     #     dif.append(total_diff)
#     f_propre_1 = []
#     f_propre_2 = []
#     f_propre_3 = []
#     for i in range(len(frequence)): 
#         f_propre_1.append(frequence[i][1])
#         f_propre_2.append(frequence[i][3])
#         f_propre_3.append(frequence[i][5])
#     fig = plt.figure()
#     ax = fig.add_subplot(111) 
#     ax.plot(number_element, f_propre_1, label = "2st frequency") 
#     ax.plot(number_element, f_propre_2, label = "4nd frequency")
#     ax.plot(number_element, f_propre_3, label = "6rd frequency")
#     ax.axvline(x=93, color='red', linestyle='--', label='converege')
#     ax.legend(loc="upper right")
#     ax.set_xticks(number_element)
#     ax.set_xlabel('Number elements[/]') 
#     ax.set_ylabel('Frequency [Hz]')
#     plt.savefig("picture/convergence_study.pdf", bbox_inches="tight", dpi=600)   
# convergence_study() 
def plot_q_deplacement(q_deplacement,dof_list,t,titre) :
    """
    Plot les déplacements en fonction du temps
        Arguments :
            - q_deplacement : vecteur des déplacements
            - dof_list : liste des degrés de liberté
            - t : temps
            - titre : titre du graphique
        Return :
            Rien
    """
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    q_deplacement  = q_deplacement.real
    q_deplacement  = q_deplacement.T
    direction_force = - q_deplacement[index_direction_force] + q_deplacement[index_direction_force+1] # direction_force_v = [-1,1,0]
    dir_force_rot   = - q_deplacement[index_rot] + q_deplacement[index_rot+1] # direction_force_v = [-1,1,0]        
    fig_1 = plt.figure(figsize=((15,5)))
    plt.plot(t,dir_force_rot*1000)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    titre_rot= titre + ".pdf"
    plt.savefig(titre_rot, bbox_inches="tight", dpi=600)
    fig_2 = plt.figure(figsize=((15,5)))
    plt.plot(t,direction_force*1000)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    titre_force = titre + "_f.pdf"
    plt.savefig(titre_force, bbox_inches="tight", dpi=600)

def deformotion_frequence_propre(X,nMode,nodes,elements) :
    """
    Plot les modes de vibration
        Arguments :
            - X : vecteur propre
            - nMode : nombre de mode repris
            - nodes : liste des noeuds
            - elements : liste des éléments
        Return :
            Rien
    """
    for mode in range(nMode) :
        matrix_def = np.copy(X)
        matrix_def = matrix_def[:,mode]
        titre = "picture/py_def_mode_" +  str(mode+1) + ".pdf"
        fig = plt.figure(figsize=((20,100)))
        ax  = fig.add_subplot(111, projection='3d')
        for i in nodes : 
            ax.scatter(i[0]/1000, i[1]/1000, i[2]/1000,color='lightblue', marker='o',alpha=0.3)
        for j in elements : 
            x = [nodes[j[0]][0]/1000, nodes[j[1]][0]/1000]
            y = [nodes[j[0]][1]/1000, nodes[j[1]][1]/1000]
            z = [nodes[j[0]][2]/1000, nodes[j[1]][2]/1000]
            ax.plot(x, y, z,color="black")
        defo_nodes = nodes.copy()
        defo_nodes = np.array(defo_nodes)/1000
        v = 0
        for i in range(4,len(defo_nodes)) :
            v = 6 * (i-4)
            defo_nodes[i][0] += 20  * matrix_def[v].real
            defo_nodes[i][1] += 20  * matrix_def[v+1].real
            defo_nodes[i][2] += 20  * matrix_def[v+2].real
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
def comp_newR_new_R_ap(q,q_ap,dof_list,t) :
    """
    Plot les déplacements en fonction du temps
        Arguments :
            - q : vecteur des déplacements
            - q_ap : vecteur des déplacements approximé
            - dof_list : liste des degrés de liberté
            - t : temps
        Return :
            Rien
    """
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    q_deplacement   = q.real
    q_deplacement   = q_deplacement.T
    direction_force = - q_deplacement[index_direction_force] + q_deplacement[index_direction_force+1] # direction_force_v = [-1,1,0]
    dir_force_rot   = - q_deplacement[index_rot] + q_deplacement[index_rot+1] # direction_force_v = [-1,1,0]        
    q_deplacement   = q.real
    q_deplacement   = q_deplacement.T

    q_deplacement_ap   = q_ap.real
    q_deplacement_ap   = q_deplacement_ap.T
    direction_force_ap = - q_deplacement_ap[0] + q_deplacement_ap[1] # direction_force_v = [-1,1,0]
    dir_force_rot_ap   = - q_deplacement_ap[4] + q_deplacement_ap[5] # direction_force_v = [-1,1,0]
    fig_1  = plt.figure(figsize=((15,5)))
    plt.plot(t,dir_force_rot*1000,label="Exact",linestyle='--',color="red")
    plt.plot(t,dir_force_rot_ap*1000,label="Approximation",alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/ap_newR_rot.pdf", bbox_inches="tight", dpi=600)
    fig_2 = plt.figure(figsize=((15,5)))
    plt.plot(t,direction_force*1000,label="Exact",linestyle='--',color="red")
    plt.plot(t,direction_force_ap*1000,label="Approximation",alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/ap_newR_force.pdf", bbox_inches="tight", dpi=600)


