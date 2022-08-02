import networkx as nx

def from_gml_to_network(path):
    '''
    Functie care returneaza un network (netorkx) dintr-un gml
    IN: path-ul catre gml.
    OUT: network-ul
    '''
    return nx.read_gml(path, label = 'id')