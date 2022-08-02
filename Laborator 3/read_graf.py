import networkx as nx
import numpy as np

def from_gml_to_network(path):
    '''
    Functie care returneaza un network (netorkx) dintr-un gml
    IN: path-ul catre gml.
    OUT: network-ul
    '''
    nx_network = nx.read_gml(path, label = 'id')
    mat = nx.adj_matrix(nx_network)
    matrix = [[0] * len(nx_network.nodes) for _ in range(len(nx_network.nodes))]
    for i in range(len(mat.nonzero()[0])):
        matrix[mat.nonzero()[0][i]][mat.nonzero()[1][i]] = 1
    net = {'noNodes': len(nx_network.nodes),
           'mat': matrix,
           'noEdges': len(nx_network.edges),
           'degrees': [degree[1] for degree in nx_network.degree()]
           }
    return net



def from_matrix_to_network(path):
    f = open(path, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(" ")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat 
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if (mat[i][j] == 1):
                d += 1
            if (j > i):
                noEdges += mat[i][j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    f.close()
    return net
    