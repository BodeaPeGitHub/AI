import sys
from math import sqrt

import networkx
import networkx as nx
import numpy as np
import tsplib95

INF = sys.maxsize


def from_gml_to_network(path):
    '''
    Functie care returneaza un network (netorkx) dintr-un gml
    IN: path-ul catre gml.
    OUT: network-ul
    '''
    nx_network = nx.read_gml(path, label='id')
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
    no_of_nodes = int(f.readline())
    matrix = []
    for i in range(no_of_nodes):
        line = f.readline()
        print(line)
        matrix.append([int(i) if int(i) != 0 else INF for i in line.split(" ")])
    source = int(f.readline()) - 1
    destination = int(f.readline()) - 1
    f.close()
    return {
        'matrix': matrix,
        'noNodes': no_of_nodes,
        'source_and_destination': [source, destination]
    }


def from_matrix_to_network_with_semicolon(path):
    f = open(path, "r")
    no_of_nodes = int(f.readline())
    matrix = []
    for i in range(no_of_nodes):
        line = f.readline()
        matrix.append([int(i) if int(i) != 0 else INF for i in line.split(",")])
    source = int(f.readline()) - 1
    destination = int(f.readline()) - 1
    f.close()
    return {
        'matrix': matrix,
        'noNodes': no_of_nodes,
        'source_and_destination': [source, destination]
    }


def from_tsp_to_network(path):
    with open(path) as file:
        line = file.readline()
        dict = {}
        while '1 ' not in line.lower():
            if 'dimension' in line.lower():
                no_of_nodes = int(line.split(" ")[-1])
            line = file.readline()
        elems = line.split(" ")
        dict[int(elems[0])] = [int(elems[1]), int(elems[2])]
        for _ in range(1, no_of_nodes):
            line = file.readline()
            elems = line.split(" ")
            dict[int(elems[0])] = [int(elems[1]), int(elems[2])]
    matrix = []
    node = 0
    for values in dict.values():
        x, y = values
        matrix.append([])
        for other_values in dict.values():
            x1, y1 = other_values
            matrix[node].append(round(sqrt((x1 - x) ** 2 + (y1 - y) ** 2)))
        node += 1
    return {
        'matrix': matrix,
        'noNodes': no_of_nodes,
        'source_and_destination': [0, 32]
    }


from_tsp_to_network('data/eil51.txt')
