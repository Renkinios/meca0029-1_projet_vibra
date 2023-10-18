import matplotlib.pyplot as plt
import functions  as fct

nodes, elements = fct.read_data('init_nodes.txt')
nodes_2, elements_2 = fct.read_data('nodes_2.txt')
def plot_wind_turbine(nodes, elements) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
    fig = plt.figure()
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
    # for node in nodes:
    #     ax.plot(node[0], node[1], node[2], 'ro')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]')
    # ax.set_zlim(0,25000)
    # fig.set_facecolor('black')
    # ax.set_facecolor('white')
    fig.set_facecolor('white')
    ax.set_facecolor('white') 
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    plt.savefig('picture/wind_turbine.pdf',bbox_inches='tight',dpi=600,format='pdf')

# plot_wind_turbine(nodes, elements) 

def plot_rigid_links(nodes, elements) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
    fig = plt.figure()
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
    # for node in nodes:
    #     ax.plot(node[0], node[1], node[2], 'ro')
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]')
    # ax.set_zlim(0,25000)
    # fig.set_facecolor('black')
    # ax.set_facecolor('white')
    fig.set_facecolor('white')
    ax.set_facecolor('white') 
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    plt.savefig('picture/rigid_links.pdf',bbox_inches='tight',dpi=600,format='pdf')
# plot_rigid_links(nodes, elements)
def plot_nodes(nodes, elements,fichier) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    main_legs = [0,1,2,3,8,9,10,11,24,25,26,27,40,41,42,43]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in main_legs : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue")
        # elif j in [56,57,58,59,60] : 
        #     x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
        #     y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
        #     z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
        #     ax.plot(x, y, z,color="green")
            # ax.scatter(nodes[21][0]/1000, nodes[21][1]/1000, nodes[21][2]/1000,color='green', marker='o')
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
    ax.set_zlabel('Z-axis [m]')
    fig.set_facecolor('white')
    ax.set_facecolor('white') 
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    # ax.set_zlim(0, 25)  
    plt.savefig(fichier,bbox_inches='tight',dpi=600,format='pdf')
# plot_nodes(nodes_2, elements_2,"picture/node_turbine_2.pdf") 
# plot_nodes(nodes, elements,"picture/node_turbine_1.pdf")
# convergence study
def convergence_study() : 
    number_element = []
    frequence = [[6.63292252E-06,6.82573769E-06,1.76861922E-05,2.44958627E-05,
    4.44303026E-05,4.76191008E-05,7.47383768E+00,7.72051594E+00],[4.38105013E-01,4.45095285E-01,7.46622518E+00,8.19863305E+00,
    1.42933489E+01,1.60522880E+01,2.14889514E+01,2.44082382E+01],[4.38105664E-01,4.45094702E-01,7.43311946E+00,8.15520812E+00,
    1.41151815E+01,1.60180064E+01,2.06556029E+01,2.32377140E+01],[4.38105716E-01,4.45094661E-01,7.42984919E+00,8.15089555E+00,
    1.40951697E+01,1.60147432E+01,2.05620338E+01,2.31014371E+01]]
    for i in range (1,len(frequence)+1) : 
        number_element.append(16*i + 28*i + 5)
    dif = [] #matrice qui va calculé le decalage par rapport au nombre d'élémént
    total_diff = 0  
    print(len(frequence)-1)
    for i in range (len(frequence)) : 
        total_diff = 0
        for j in range(8) : 
            total_diff += abs(frequence[0][j] - frequence[i][j])
        dif.append(total_diff)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(number_element, dif)
    ax.set_xlabel('Number elements[/]')
    ax.set_ylabel('The sum of the frequency differences between the first and the last iteration [Hz]')
    plt.savefig("picture/convergence_study.pdf", bbox_inches="tight", dpi=600)
    
convergence_study()
