import numpy as np
import matplotlib.pyplot as plt
import reader 

nodes, elements = reader.read_data('init_nodes.txt')
reader.plot_nodes(nodes, elements)

