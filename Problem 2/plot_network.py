import matplotlib.pyplot as plt 
import numpy as np 
import networkx as nx

def plot_network(network, communities = None):
    if communities is None:
        communities = [1] * nx.number_of_nodes(network)
    np.random.seed(123) 
    pos = nx.spring_layout(network)
    plt.figure(figsize=(4, 4)) 
    nx.draw_networkx_nodes(network, pos, node_size = 75, cmap=plt.cm.RdYlBu, node_color = communities)
    nx.draw_networkx_edges(network, pos, alpha=0.3)
    plt.show()