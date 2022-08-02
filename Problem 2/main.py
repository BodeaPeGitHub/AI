from operator import ne
from telnetlib import GA
from gml_to_networkx import from_gml_to_network
from plot_network import plot_network
from girvan_newman import girvan_newman
import networkx as nx
import sys
from datetime import datetime


def main_program():
    # network = from_gml_to_network("data/dolphins/dolphins.gml") 
    # network_copy = from_gml_to_network("data/dolphins/dolphins.gml") 
    # plot_network(network)
    # communities = girvan_newman(network_copy, 2)
    # plot_network(network, communities)
    time_start = datetime.now()
    test_communities_equality("data/dolphins/dolphins.gml", 2)
    test_communities_equality("data/football/football.gml", 2) # - Sper. aici e de fapt 12
    test_communities_equality("data/karate/karate.gml", 2) 
    test_communities_equality("data/lesmis/lesmis.gml", 2) # 1 
    test_communities_equality("data/polbooks/polbooks.gml", 2) # 2 
    test_communities_equality("data/netscience/netscience.gml", 397) # 3 
    test_communities_equality("data/power/power.gml", 25) # 4 
    test_communities_equality("data/adjnoun/adjnoun.gml", 2) # 5 
    test_communities_equality("data/as-22july06/as-22july06.gml", 2)
    test_communities_equality("data/map/map.gml", 2) # 6
    test_communities_equality("data/krebs/krebs.gml", 2) # - aici e de fapt 3
    time_finish = datetime.now()
    print("Totul merge ca pe roate, dar merge in ", time_finish - time_start, "secunde.")


def greedyCommunitiesDetectionByTool(path):
    from networkx.algorithms import community
    G = from_gml_to_network(path)
    communities_generator = community.girvan_newman(G)
    return list(sorted(c) for c in next(communities_generator))


def write_to_file(communities, path):
    name = path.split('/')[-1]
    print(name)
    with open('solutii/' +  name, 'w+') as file:
        stdout = sys.stdout
        sys.stdout = file
        index = 1
        for elem in communities:
            print(index, elem) 
            index += 1
        sys.stdout = stdout

def test_communities_equality(path, size):
    communities = greedyCommunitiesDetectionByTool(path)
    network = from_gml_to_network(path)
    print(len(communities))
    com = girvan_newman(network, size)
    write_to_file(com, path)
    print(communities)
    plot_network(from_gml_to_network(path), com[:-1])
    for elem in nx.connected_components(network):
        print(elem)
        assert(sorted(list(elem)) in communities)



if __name__ == "__main__":
    main_program()