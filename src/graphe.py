import matplotlib.pyplot as plt
import functions         as fct
import numpy             as np
import write_read        as write_read
import methode           as mth
import matrice           as mtx
import time
from mpl_toolkits.mplot3d import Axes3D
def plot_nodes(nodes, elements,fichier,leg_elem, rili_elem) : 
    """ Plot la structure avec les noeuds et les éléments
        Arguments : 
            - noeud : liste des noeuds
            - elements : liste des éléments
        Return : 
            - Rien
    """
    fig = plt.figure(figsize=((15,8)))
    ax = fig.add_subplot(111, projection='3d')
    j = 0 
    for i in elements:
        if j in leg_elem : 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="red",linewidth=0.8)
        elif j in rili_elem: 
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="green",linewidth=0.8)
            ax.scatter(nodes[21][0]/1000, nodes[21][1]/1000, nodes[21][2]/1000,color='green', marker='o',linewidth=0.8)
        else :
            x = [nodes[i[0]][0]/1000, nodes[i[1]][0]/1000]
            y = [nodes[i[0]][1]/1000, nodes[i[1]][1]/1000]
            z = [nodes[i[0]][2]/1000, nodes[i[1]][2]/1000]
            ax.plot(x, y, z,color="blue",linewidth=0.8)
        j += 1
    # for node in nodes:
    #     ax.plot(node[0]/1000, node[1]/1000, node[2]/1000, 'peru', marker='o', markersize = 3)
    ax.set_xlabel('X-axis [m]')
    ax.set_ylabel('Y-axis [m]')
    ax.set_zlabel('Z-axis [m]', labelpad=12)
    ax.invert_xaxis()
    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_box_aspect([1, 1, 4])
    ax.grid(False)
    plt.savefig(fichier,bbox_inches='tight',dpi=600,format='pdf')
    plt.close()
def conv_nx() : 
    frequence = [
    [4.42401783E-01, 4.49817787E-01, 9.60608860E-01, 6.89354897E+00, 7.32969389E+00, 1.63825977E+01, 2.04311445E+01, 2.22578560E+01],
    [4.42402461E-01, 4.49817194E-01, 9.60603890E-01, 6.87080562E+00, 7.30317814E+00, 1.63451574E+01, 1.97419262E+01, 2.14418084E+01],
    [4.42402506E-01, 4.49817158E-01, 9.60603607E-01, 6.86900918E+00, 7.30109435E+00, 1.63422428E+01, 1.96804680E+01, 2.13663317E+01],
    [4.42402506E-01, 4.49817158E-01, 9.60603607E-01, 6.86900918E+00, 7.30109435E+00, 1.63422428E+01, 1.96804680E+01, 2.13663317E+01]
]
    relatif_error = []
    for i in range(4) :
        tot_rel_error = 0
        for j in range(8) :
            tot_rel_error =+ (abs(frequence[i][j] - frequence[3][j])/frequence[3][j])
        relatif_error.append(tot_rel_error/8)
    X = [1,2,3,4]

    plt.figure(figsize=((15,5)))
    plt.plot(X,relatif_error)
    plt.xlabel("number of elements per beam")
    plt.ylabel("relative error")
    plt.xticks([1,2,3,4])
    plt.savefig("picture/conv_nx.pdf", bbox_inches="tight", dpi=600)   
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
    direction_force = (- q_deplacement[index_direction_force] + q_deplacement[index_direction_force+1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_force_rot   = (- q_deplacement[index_rot] + q_deplacement[index_rot+1])/np.sqrt(2) # direction_force_v = [-1,1,0]        
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
    plt.close()

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
        fig = plt.figure(figsize=((5,10))) 
        ax  = fig.add_subplot(111, projection='3d')
        for j in elements : 
            x = [nodes[j[0]][0]/1000, nodes[j[1]][0]/1000]
            y = [nodes[j[0]][1]/1000, nodes[j[1]][1]/1000]
            z = [nodes[j[0]][2]/1000, nodes[j[1]][2]/1000]
            ax.plot(x, y, z,color="black",linewidth=0.8)
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
            ax.plot(x, y, z,color="red",linewidth=0.8)
        ax.set_xlabel('X-axis [m]')
        ax.set_ylabel('Y-axis [m]')
        ax.set_zlabel('Z-axis [m]')
        ax.set_zlabel('Z-axis [m]', labelpad=100)
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_box_aspect([1, 1, 4])
        ax.grid(False)
        plt.savefig(titre, bbox_inches="tight", dpi=600)
        plt.close()
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
    direction_force = (- q_deplacement[index_direction_force] + q_deplacement[index_direction_force+1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_force_rot   = (- q_deplacement[index_rot] + q_deplacement[index_rot+1])/np.sqrt(2) # direction_forcedirection_force_v = [-1,1,0]        
    q_deplacement   = q.real
    q_deplacement   = q_deplacement.T

    q_deplacement_ap   = q_ap.real
    q_deplacement_ap   = q_deplacement_ap.T
    direction_force_ap = (- q_deplacement_ap[0] + q_deplacement_ap[1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_force_rot_ap   = (- q_deplacement_ap[4] + q_deplacement_ap[5])/np.sqrt(2) # direction_force_direction_force_v = [-1,1,0]
    fig_1  = plt.figure(figsize=((15,5)))
    plt.plot(t,dir_force_rot*1000,label="Exact",linestyle='--',color="red")
    plt.plot(t,dir_force_rot_ap*1000,label="Approximation",alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/ap_newR_rot.pdf", bbox_inches="tight", dpi=600)
    plt.close()

    fig_2 = plt.figure(figsize=((15,5)))
    plt.plot(t,direction_force*1000,label="Exact",linestyle='--',color="red")
    plt.plot(t,direction_force_ap*1000,label="Approximation",alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/ap_newR_force.pdf", bbox_inches="tight", dpi=600)
    plt.close()

def comp_depl_acc_newR(q_new,q_dep,q_acc,t,dof_list):
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    q_dep     = q_dep.real
    q_dep     = q_dep.T
    dir_f_dep = (- q_dep[index_direction_force] + q_dep[index_direction_force+1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_r_dep = (- q_dep[index_rot] + q_dep[index_rot+1])/np.sqrt(2) #
    q_acc     = q_acc.real
    q_acc     = q_acc.T
    dir_f_acc = (- q_acc[index_direction_force] + q_acc[index_direction_force+1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_r_acc = (- q_acc[index_rot] + q_acc[index_rot+1])/np.sqrt(2) 
    q_new     = q_new.real
    q_new     = q_new.T
    dir_f_new = (- q_new[index_direction_force] + q_new[index_direction_force+1])/np.sqrt(2) 
    dir_r_new = (- q_new[index_rot] + q_new[index_rot+1])/np.sqrt(2) 
    fig_1 = plt.figure(figsize=((15,5)))
    plt.plot(t,dir_f_dep*1000,label="Deplacement",linestyle='--',color="red")
    plt.plot(t,dir_f_acc*1000,label="Acceleration",alpha=0.7)
    plt.plot(t,dir_f_new*1000,label="Newmark",alpha=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/comp_depl_acc_newR_force.pdf", bbox_inches="tight", dpi=600)
    plt.close()

    fig_2 = plt.figure(figsize=((15,5)))
    plt.plot(t,dir_r_dep*1000,label="Deplacement",linestyle='--',color="red",alpha=0.7)
    plt.plot(t,dir_r_acc*1000,label="Acceleration",linestyle = ':' )
    plt.plot(t,dir_r_new*1000,label="Newmark",alpha=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig("picture/comp_depl_acc_newR_rot.pdf", bbox_inches="tight", dpi=600)
    plt.close()

def conp_Mode_dep(M,K,w,x,eps,p,t,dof_list,nMode=8,rotor=False) : 
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    plt.figure(figsize=((15,5)))
    titre = "picture/comp_dep_mode_"
    if rotor :
        titre += "rotor"
    else :
        titre += "force"
    for i in range(1,nMode) :
        if rotor :
            q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,i)
            q_dep     = q_deplacement.real
            q_dep     = q_dep.T
            dir_r_dep = (- q_dep[index_rot] + q_dep[index_rot+1])/np.sqrt(2)
            plt.plot(t,dir_r_dep*1000,label="Mode "+ str(i))
        else :
            q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,i)
            q_dep     = q_deplacement.real
            q_dep     = q_dep.T
            dir_f_dep = (- q_dep[index_direction_force] + q_dep[index_direction_force+1])/np.sqrt(2)
            plt.plot(t,dir_f_dep*1000,label="Mode "+ str(i))
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig(titre +".pdf", bbox_inches="tight", dpi=600)
    plt.close()

def conp_Mode_acc(M,K,w,x,eps,p,t,dof_list,nMode=8,rotor=False) : 
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    plt.figure(figsize=((15,5)))
    titre = "picture/comp_acc_mode_"
    if rotor :
        titre += "rotor"
    else :
        titre += "force"
    for i in range(1,nMode) :
        if rotor :
            q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,i)
            q_acc     = q_acc.real
            q_acc     = q_acc.T
            dir_r_dep = (- q_acc[index_rot] + q_acc[index_rot+1])/np.sqrt(2)
            plt.plot(t,dir_r_dep*1000,label="Mode "+ str(i))
        else :
            q_deplacement, q_acc = mth.methode_superposition(M,K,w,x,eps,p,t,i)
            q_acc     = q_acc.real
            q_acc     = q_acc.T
            dir_f_dep = (- q_acc[index_direction_force] + q_acc[index_direction_force+1])/np.sqrt(2)
            plt.plot(t,dir_f_dep*1000,label="Mode "+ str(i))
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig(titre +".pdf", bbox_inches="tight", dpi=600)
    plt.close()

def conv_time_new(t,M,C,K,dof_list,rotor = False) :
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    plt.figure(figsize=((15,5)))
    titre = "picture/comp_time_new"
    if rotor : 
        titre += "_rotor"
    else :
        titre += "_force"
    for i in [100,500,1000,2000] : 
        t      = np.linspace(0, 10, i)
        p      = mtx.force_p(M,dof_list,t) 
        q      = mth.New_mth(t,M,C,K,p)
        q      = q.real
        q      = q.T
        if rotor : 
            dir_f_dep = (- q[index_rot] + q[index_rot+1])/np.sqrt(2)
            plt.plot(t,dir_f_dep*1000,label="Delta "+ str(10/i))
        else :
            dir_f_dep = (- q[index_direction_force] + q[index_direction_force+1])/np.sqrt(2)
            plt.plot(t,dir_f_dep*1000,label="Delta "+ str(10/i) + "[s]")
    plt.xlabel("Time [s]")
    plt.ylabel("Desplacement [mm]")
    plt.legend(loc="upper right")
    plt.savefig(titre +".pdf", bbox_inches="tight", dpi=600)
    plt.close()

def comp_Craig_guyan(Mcc,Kcc,Krr,Rgi,Kt,Mt,w_gi,Neigenmodes,nMode,w) :
    plt.figure(figsize=((10,8)))
    x = np.linspace(1,8,8)
    plt.scatter(x,w_gi/(2*np.pi),label="guyan_irons [Hz]",marker="*")
    plt.scatter(x,w/(2*np.pi),label="Exact frequency [Hz]",marker="x")
    lab = "Craig_Bampton Mode"
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    for i in  range(1,Neigenmodes+1) : 
        K_cb, M_cb, Rcb  = mth.Craig_Bampton(Mcc,Kcc,Krr,Rgi,Neigenmodes,nMode,Kt,Mt)
        w_cb, x_cb  = fct.natural_frequency(M_cb, K_cb,nMode)
        plt.scatter(x,w_cb/(2*np.pi),label= lab + str(i) ,marker="o", facecolors='none',edgecolors=couleurs[i-1])
    plt.ylim(bottom=0, top=27)
    plt.xlabel("Frequency number")
    plt.ylabel("Frequence [Hz]")
    plt.legend(loc="upper left")
    plt.savefig("picture/comp_f_Craig_guyan.pdf", bbox_inches="tight", dpi=600)
    plt.close()

def fft_new_R(q_new,t,dof_list) : 
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    q_new     = q_new.real
    q_new     = q_new.T
    dir_f_new = (- q_new[index_direction_force] + q_new[index_direction_force+1])/np.sqrt(2)
    dir_r_new = (- q_new[index_rot] + q_new[index_rot+1])/np.sqrt(2)
    
    frequencies = 1 * np.arange(0, len(t) // 2) / len(t) / (t[1] - t[0])
    
    fft_force = np.fft.fft(dir_f_new)
    F_impact_disp = np.abs(fft_force / len(t))
    F_impact_disp = F_impact_disp[:len(t) // 2]
    F_impact_disp[1:-1] = 2 * F_impact_disp[1:-1]

    fft_rotor = np.fft.fft(dir_r_new)
    F_rotor = np.abs(fft_rotor / len(t))
    F_rotor = F_rotor[:len(t) // 2]
    F_rotor[1:-1] = 2 * F_rotor[1:-1]

    # Tracé des résultats
    plt.figure(figsize=(15, 5))
    plt.semilogy(frequencies, F_impact_disp*1000)
    plt.xlim([0, 25])
    plt.xticks([0.4,1,7,10,15,20,25])

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Desplacement [mm]")
    plt.savefig("picture/fft_newR_force.pdf", bbox_inches="tight", dpi=600)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.semilogy(frequencies, F_rotor*1000)
    plt.xlim([0, 25])
    plt.xlabel("Frequency [Hz]")
    plt.xticks([0.4,1,5,10,15,20,25])

    plt.ylabel("Amplitude [mm]")
    plt.savefig("picture/fft_newR_rot.pdf", bbox_inches="tight", dpi=600)
    plt.close()

def comp_accurancy_time(q,Mcc,Kcc,Krr,Rgi,Neigenmodes,nMode,Kt,Mt,p_t,C_t,t,dof_list) : 
    index_rot = dof_list[21][0] - 1 - 6 * 4 # en x
    index_direction_force = dof_list[17][0]- 1 - 6 * 4 # en x
    q_deplacement   = q.real
    q_deplacement   = q_deplacement.T
    direction_force = (- q_deplacement[index_direction_force] + q_deplacement[index_direction_force+1])/np.sqrt(2) # direction_force_v = [-1,1,0]
    dir_force_rot   = (- q_deplacement[index_rot] + q_deplacement[index_rot+1])/np.sqrt(2) 
    q_deplacement   = q_deplacement.T

    matrix_t        = []
    error_force     = []
    error_rot       = []
    for i in range(Neigenmodes) :
        K_cb, M_cb, Rcb  = mth.Craig_Bampton(Mcc,Kcc,Krr,Rgi,Neigenmodes,nMode,Kt,Mt)
        C_new            = Rcb.T @ C_t @ Rcb
        p_new            = Rcb.T @ p_t
        t_app = time.time()
        q_ap = mth.New_mth(t,M_cb,C_new,K_cb,p_new)
        t_end = time.time()
        delta_t = t_end - t_app
        matrix_t.append(delta_t)
        q_deplacement_ap     = q_ap.real
        q_deplacement_ap     = q_deplacement_ap.T
        direction_force_ap   = (- q_deplacement_ap[0] + q_deplacement_ap[1])/np.sqrt(2) # direction_force_v = [-1,1,0]
        dir_force_rot_ap     = (- q_deplacement_ap[4] + q_deplacement_ap[5])/np.sqrt(2) # direction_force_
        MAE_F = 0 
        MAE_R = 0
        for i in (range(len(direction_force_ap))) : 
            MAE_F += (direction_force_ap[i]-direction_force[i])
            MAE_R += (dir_force_rot_ap[i]-dir_force_rot[i])
        error_force.append(MAE_F/len(direction_force_ap))
        error_rot.append(MAE_R/len(dir_force_rot_ap))

    error_force = np.array(error_force)
    error_rot   = np.array(error_rot)
    delta_t     = np.array(matrix_t)
    plt.figure(figsize=((15,5)))
    plt.plot(delta_t,error_force)
    plt.xlabel("computation time [s]")
    plt.ylabel("relative error [%]")
    plt.savefig("picture/comp_time_force.pdf", bbox_inches="tight", dpi=600)
    plt.close()


    