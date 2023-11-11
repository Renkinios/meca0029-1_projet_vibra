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