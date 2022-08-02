import networkx as nx

def girvan_newman(network, number_of_communities = 2):
    '''
    Functie care imparte un graf intr-un numar dat de comunitati.
    IN: graful (care este un graf de tipul networkx) si numarul de comunitati (int) care este defaul 2 
    OUT: o lista unde pe fiecare poztiei este trcuta culoare nodului.
    '''
    while number_of_communities != len(list(nx.connected_components(network))):
        betweenes_centrality_of_all_edges = nx.edge_betweenness_centrality(network)
        x, y = find_max_in_dictionary(betweenes_centrality_of_all_edges)
        network.remove_edge(x, y)

    components = [0] * (nx.number_of_nodes(network) + 1)
    index = 0
    for component in nx.connected_components(network):
        index += 1
        for node in component:
            components[node] = index
    return components


def find_max_in_dictionary(dict):
    '''
    Functie care returneaza cheia pentru care valoarea este maxima intr-un dictionar.
    IN: dictionarul
    OUT: cheia pentru care valoare este maxima
    '''
    maximum = 0
    max_key = None
    for key, value in dict.items():
        if value > maximum:
            max_key = key
            maximum = value 
    return max_key