import functions as fct
nodes, elements = fct.read_data('Data/init_nodes.txt')

new_nodes, new_element, new_leg, new_rili  = fct.new_nodes(nodes,elements,3)