import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from read_graf import INF


def create_colors(network, path):
    edge_labels = {}
    edge_colors = ['k'] * nx.number_of_edges(network)
    path = [(x - 1, y - 1) if x < y else (y - 1, x - 1) for x, y in zip(path[:-1], path[1:])]
    edges = list(nx.edges(network))
    for index in range(len(edges)):
        if edges[index] in path:
            edge_colors[index] = 'r'
            x, y = edges[index]
            edge_labels[edges[index]] = network.edges[x, y]['weight']
    return edge_colors, edge_labels


def create_graf(network):
    nx_network = nx.Graph()
    nx_network.add_nodes_from([i for i in range(network['noNodes'])])
    for index in range(len(network['matrix'])):
        for column in range(len(network['matrix'][index])):
            if network['matrix'][index][column] != INF:
                nx_network.add_edge(index, column, weight=network['matrix'][index][column])
    return nx_network


def plot_network(network, name="", path=None):
    if path is None:
        path = []
    nx_network = create_graf(network)
    pos = nx.spring_layout(nx_network, k=0.9)
    plt.figure(figsize=(7, 7))
    plt.title(name, fontsize=18)
    edge_colors, edge_labels = create_colors(nx_network, path)
    nx.draw_networkx_nodes(nx_network, pos, node_size=75, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_edges(nx_network, pos, alpha=0.3)
    nx.draw_networkx_edge_labels(nx_network, pos, edge_labels=edge_labels, font_size=15)
    names = {}
    for i in range(network['noNodes']):
        names[i] = i + 1
    nx.draw(nx_network, pos, with_labels=True, labels=names, edge_color=edge_colors)
    plt.show()
