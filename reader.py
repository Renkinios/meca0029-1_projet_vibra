# Initialiser la nodes
import numpy as np
import matplotlib.pyplot as plt
nodes = []
elements = []
# Ouvrir le fichier en mode lecture
with open('init_nodes.txt', 'r') as fichier:
    lignes = fichier.readlines()

# Ignorer la première ligne (Number of nodes)
lignes = lignes[1:]
cumpt = 0 
# Parcourir les lignes restantes et insérer les coordonnées dans la nodes
for ligne in lignes:
    # Séparer la ligne en tokens en utilisant l'espace comme délimiteur
    if("Number of elements :\n" == ligne):
        break
    tokens = ligne.split()
    cumpt += 1
    coordonnees = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
    nodes.append(coordonnees)
for ligne in range(cumpt+1, len(lignes)):
    # Séparer la ligne en tokens en utilisant l'espace comme délimiteur
    tokens = lignes[ligne].split()
    coordonnees = [int(tokens[2]), int(tokens[3])]
    elements.append(coordonnees)

# print("éléments : ", elements)
# print("nodes : ", nodes)
print("coordonnees :",nodes[elements[30][0]][0])
def new_nodes(matrice):

    new_matrix = []
    for i in range(len(matrice)):
        x = (nodes[matrice[i][0]][0] + nodes[matrice[i][1]][0])/2
        y = (nodes[matrice[i][0]][1] + nodes[matrice[i][1]][1])/2
        z = (nodes[matrice[i][0]][2] + nodes[matrice[i][1]][2])/2
        nodes.append([x, y, z])
        new_element_1 = [matrice[i][0], len(nodes)-1]
        new_element_2 = [len(nodes)-1, matrice[i][1]]
        new_matrix.append(new_element_1)
        new_matrix.append(new_element_2)

    return new_matrix
elements = new_nodes(elements)
def writing_nodes_element_file(nodes,elements):
    with open('new_nodes.txt', 'w') as fichier:
        fichier.write("Number of nodes " + str(len(nodes)) + " :\n")
        for i in range(len(nodes)):
            fichier.write("\t" + str(i) + " : " + str(nodes[i][0]) + " " + str(nodes[i][1]) + " " + str(nodes[i][2]) + "\n")
        fichier.write("Number of elements :\n")
        for i in range(len(elements)):
            fichier.write("\t"+ str(i) + " : " + str(elements[i][0]) + " " + str(elements[i][1]) + "\n")
writing_nodes_element_file(nodes,elements)
def plot_nodes(noeud, elements) : 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in elements:
        x = [noeud[i[0]][0], noeud[i[1]][0]]
        y = [noeud[i[0]][1], noeud[i[1]][1]]
        z = [noeud[i[0]][2], noeud[i[1]][2]]
        ax.plot(x, y, z, 'b-')
    for node in nodes:
        ax.plot(node[0], node[1], node[2], 'ro')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Maillage de droites en 3D')
    ax.set_zlim(0,25000)
    plt.show()
plot_nodes(nodes, elements)
