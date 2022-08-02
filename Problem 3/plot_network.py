import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def plot_network(network, communities=None, name=""):
    if communities is None:
        communities = [1] * network['noNodes']
    np.random.seed(123)
    nx_network = nx.from_numpy_matrix(np.matrix(network['mat']))
    pos = nx.spring_layout(nx_network)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    nx.draw_networkx_nodes(nx_network, pos, node_size=75, cmap=plt.cm.RdYlBu, node_color=communities)
    nx.draw_networkx_edges(nx_network, pos, alpha=0.3)
    plt.show()
