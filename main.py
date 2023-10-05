import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math

# Visualisation de la structure initiale
nodes, elements = fct.read_data('init_nodes.txt')
fct.plot_nodes(nodes, elements)
# print(fct.euclidian_distance(6, elements, nodes))

# Modification du nombre d'éléments
# elements = fct.new_nodes(nodes, elements)
# fct.writing_nodes_element_file(nodes, elements, 'nodes_2.txt')
# nodes, elements = fct.read_data('nodes_2.txt')
# fct.plot_nodes(nodes, elements)

# Création de la liste des degrés de liberté
count = 1
dof_list = []
for i in range(len(nodes)) : 
    dof_elem = [count, count+1, count+2, count+3, count+4, count+5]
    dof_list.append(dof_elem)
    count+=6


# Création de la matrice locel (reprenant les dof impliqués pour chaque élément)
locel = []
for i in range(len(elements)) :    
    locel.append([dof_list[elements[i][0]][0],dof_list[elements[i][0]][1], dof_list[elements[i][0]][2], dof_list[elements[i][0]][3], dof_list[elements[i][0]][4], dof_list[elements[i][0]][5],
                  dof_list[elements[i][1]][0], dof_list[elements[i][1]][1], dof_list[elements[i][1]][2], dof_list[elements[i][1]][3], dof_list[elements[i][1]][4], dof_list[elements[i][1]][5]])


# Définition des constantes pour la structure
E = 210e6                                           # Module de Young [Pa]
A_leg = math.pi*(1 - (1-0.02)**2)                   # Section d'une poutre principale [m^2]
A_beam = math.pi*(0.6**2 - (0.6-0.02)**2)           # Section d'une poutre secondaire [m¨2]
nu = 0.3                                            # Coefficient de Poisson [-]
rho = 7800                                          # Densité de l'acier utilisé [kg/m^3]
G = E/(2*(1+nu))                                    # Module de cisaillement [Pa]
h = 1

# Définition des caractéristiques pour une poutre secondaire (selon les axes locaux)
m_beam = rho*math.pi*h*(0.6**2 - (O.6-0.02)**2)                    # Masse [kg]
Iyz_beam = (math.pi/4)*(0.6**4 - (0.6-0.02)**2)                    # Moment quadratique selon l'axe y et z [m^4]
Ix_beam = (math.pi/2)*(0.6**4 - (0.6-0.02)**2)                     # Moment quadratique selon l'axe x [m^4]
Jx_beam = 0.5*m_beam*(O.6**2 + (0.6-0.02)**2)                      # Moment d'inertie selon l'axe x [kg.m^2]
Jyz_beam = 0.25*m_beam*(O.6**2 + (0.6-0.02)**2)+(m_beam*(h**2))/12 # Moment d'inertie selon l'axe y et z [km.m^2]

# Définition des caractéristiques pour une pour principale (elon les axes locaux)
m_leg = rho*math.pi*h*(1**2 - (1-0.02)**2)                    # Masse d'une poutre principale [kg]
Iyz_leg = (math.pi/4)*(1**4 - (1-0.02)**2)                    # Moment quadratique selon l'axe y et z [m^4]
Ix_leg = (math.pi/2)*(1**4 - (1-0.02)**2)                     # Moment quadratique selon l'axe x [m^4]
Jx_leg = 0.5*m_beam*(1**2 + (1-0.02)**2)                      # Moment d'inertie selon l'axe x [kg.m^2]
Jyz_leg = 0.25*m_beam*(1**2 + (1-0.02)**2)+(m_beam*(h**2))/12 # Moment d'inertie selon l'axe y et z [kg.m^2]

# Définition des constantes pour les rigid links
rho_r = rho*1e-4    # Densité [kg/m^3]
E_R = E*1e4         # Module de Young [Pa]
A_r = A_leg*1e-2    # Section [m^2]
Iyz_r = Iyz_leg*1e4 # Moment quadratique selon l'axe y et z [m^4]
Jx_r = Jx_leg*1e4   # Moment d'intertie selon l'axe x [kg.m^2]


