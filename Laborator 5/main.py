from ACO import ACO
from Graph import Graph
from plot import plot_graph
from read_graf import from_tsp_to_network


def main():
    network = from_tsp_to_network('data/eil51.tsp')
    plot_graph(network['points'], [])
    graph = Graph(network)
    problem_parameters = dict(rho=0.3, beta=10.0, alpha=1.0, generations=100, ant_count=network['dimension'], q=3, dinamic=0, graph=graph)
    aco = ACO(problem_parameters)
    path, cost = aco.solve()
    print('cost: {}, path: {}'.format(cost, path))
    plot_graph(network['points'], path)


if __name__ == '__main__':
    main()
