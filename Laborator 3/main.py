import networkx as nx
import numpy as np
from genetic_algorithm import GA
from plot_network import plot_network
from read_files import read_files
from read_graf import from_gml_to_network, from_matrix_to_network
from girvan_newman import greedy_communities_detection_by_tool
from random import seed
import warnings

warnings.simplefilter('ignore')

def modularity_density(communities, param, resolution_parameter=0.5):
    '''
    Modulraity density function.
    Functia este bazata pe functia de modularitate dar tine cont si de retele mai mici din cadrul unei relete mai mari de care modularity nu tine cont din cauza scorului prea mare.
    '''
    Q = 0
    G = nx.from_numpy_matrix(np.matrix(param['mat']))
    com = {}
    for i in range(len(communities)):
        if communities[i] in com:
            com[communities[i]].append(i)
        else:
            com[communities[i]] = []
    communities = []
    for value in com.values():
        print(value)
        communities.append(value)
    for community in communities:
        sub = nx.subgraph(G, community)
        sub_n = sub.number_of_nodes()
        dint = []
        dext = []
        for node in sub:
            dint.append(sub.degree(node))
            dext.append(G.degree(node) - sub.degree(node))

        try:
            Q += (1 / sub_n) * (
                        (2 * resolution_parameter * np.sum(dint)) - (2 * (1 - resolution_parameter) * np.sum(dext)))
        except ZeroDivisionError:
            pass
    return Q


def z_modularity(communities, param):
    '''
    Modulraity Z - function.
    Functia este bazata pe functia de modularitate dar tine cont si de retele mai mici din cadrul unei relete mai mari de care modularity nu tine cont din cauza scorului prea mare.
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147805
    '''
    Q = 0
    G = nx.from_numpy_matrix(np.matrix(param['mat']))
    com = {}
    for i in range(len(communities)):
        if communities[i] in com:
            com[communities[i]].append(i)
        else:
            com[communities[i]] = []
    communities = []
    for value in com.values():
        communities.append(value)
    m = G.number_of_edges()
    mmc = 0
    dc2m = 0
    for community in communities:
        sub = nx.subgraph(G, community)
        sub_n = sub.number_of_nodes()
        dc = 0
        for node in sub:
            dc += G.degree(node)

        mmc = sub_n / m
        dc2m += (dc / (2 * m)) ** 2
    res = 0
    try:
        res = (mmc - dc2m) / np.sqrt(dc2m * (1 - dc2m))
    except ZeroDivisionError:
        pass
    return res


def modularity(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['noEdges']
    M = 2 * noEdges
    Q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if communities[i] == communities[j]:
                Q += (mat[i][j] - degrees[i] * degrees[j] / M)
    return Q * 1 / M


def menu():
    data = read_files('data/paths.txt')
    for elems in data:
        print(elems['name'])
        maxs = find_max_in_generation(elems)
        test(elems, maxs)


def find_max_in_generation(data):
    network = from_gml_to_network(data['path'])
    ga_params = dict(popSize=150, noGen=100)
    problem_params = network
    problem_params['noCommunities'] = data['noOfCommunities']
    problem_params['function'] = modularity
    problem_params['function'] = modularity_density
    # problem_params['function'] = z_modularity
    all_best_chromosomes = []
    ga = GA(ga_params, problem_params)
    ga.initialisation()
    ga.evaluation()
    generations = []
    for generation in range(ga_params['noGen']):
        all_best_chromosomes.append(ga.bestChromosome())
        generations.append(generation)
        ga.oneGeneration()
        # ga.oneGenerationElitism()
        # ga.oneGenerationSteadyState()
    return all_best_chromosomes


def test(data, all_best_chromosomes):
    network = from_gml_to_network(data['path'])
    communities = greedy_communities_detection_by_tool(network)
    generation = 0
    maxim = all_best_chromosomes[0]
    max_gen = 0
    for chromosome in all_best_chromosomes:
        if maxim.fitness < chromosome.fitness:
            maxim = chromosome
            max_gen = generation
        if chromosome.representation == communities:
            print('Found answer in generation', generation)
        else:
            diff = 0
            for x, y in zip(communities, chromosome.representation):
                if x != y:
                    diff += 1
            print('Found', diff, 'diferences in generation', generation, 'with modularity', chromosome.fitness)
        generation += 1
    plot_network(network, maxim.representation,
                 data['name'] + ' generation ' + str(max_gen) + '\nModularity ' + str(maxim.fitness))


menu()
