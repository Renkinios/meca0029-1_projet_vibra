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
def write_results(f,masse_totals,eps,temps_new,f_gi,t_gi,f_cg,t_cg,t_app_new,c_time):
    """ Ecrit les resultats dans un fichier texte
        Arguments : 
            - f : liste des frequence naturelles
            - masse_totals : masse total de la structure
            - eps : liste des facteurs d'amortissement
            - temps_new : temps de calcul de la methode de Newmark
            - f_gi : liste des frequence naturelles de la methode de Guyan-Irons
            - t_gi : temps de calcul de la methode de Guyan-Irons
            - f_cg : liste des frequence naturelles de la methode de Craig-Bampton
            - t_cg : temps de calcul de la methode de Craig-Bampton
            - t_app_new : temps de calcul de la methode de Newmark approcimate par Craig-Bampton
            - c_time : si on veux ecrire le temps 
        Return : 
            - Rien
    """
    with open('Data/results.txt', 'w') as fichier:
        
        fichier.write("Natural frequency : \n")
        for i in range(len(f)):
            fichier.write("Mode" + str(i) + " :\t " + str(round(f[i],8))[:8] + " [Hz] \n")
        
        fichier.write("\nNatutal frequency of guyan-irons : \n")
        for i in range(len(f_gi)):
            fichier.write("Mode" + str(i) + " :\t" + str(round(f_gi[i],8))[:8] + " [Hz] " + "\tRelative error : " + str(np.abs(f_gi[i]-f[i])/f[i]*100) + " [%] \n")
        
        fichier.write("\nNatutal frequency of Craig-Bampton : \n")
        for i in range(len(f_cg)):
            fichier.write("Mode" + str(i) + " :\t " + str(round(f_cg[i],8))[:8] + " [Hz] \t" + "Relative error : " + str(np.abs(f_cg[i]-f[i])/f[i]*100) + " [%] \n")

        fichier.write("\nMasse total :" + str(masse_totals) + "[kg] \n")
        
        fichier.write("\nDamping ratio :\n")
        for i in range(len(eps)):
            fichier.write("Mode" + str(i) + ": \t" + str(eps[i]) + " \n")
        if c_time : 
            fichier.write("\nTime of application The Newmark Method: " + str(round(temps_new,4)) + " [s] \n")
            fichier.write("Time of application The Newark Method approcimate by Craig-Bampton: " + str(t_app_new) + " [s] \n")
            fichier.write("Time of application The Guyan-Irons Method: " + str(round(t_gi,4)) + " [s] \n")
            fichier.write("Time of application The Craig-Bampton Method: " + str(round(t_cg,4)) + " [s] \n")