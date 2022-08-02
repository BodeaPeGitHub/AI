from random import randint, uniform
from Ant import Ant


class ACO:

    def __init__(self, problem_parameters: dict):
        self.rho = problem_parameters['rho']  # coeficientul rezidual (evaporare)
        self.beta = problem_parameters['beta']  # importanta vizibilitati
        self.alpha = problem_parameters['alpha']  # importanta feromonului
        self.generations = problem_parameters['generations']
        self.ant_count = problem_parameters['ant_count']
        self.Q = problem_parameters['q']  # intensitatea feromonului
        self.__dinamic = problem_parameters['dinamic']
        self.__graph = problem_parameters['graph']
        self.__lowest, self.__highest = self.__find_lowest_highest()

    def __update_pheromone(self, ants: list):
        for i, row in enumerate(self.__graph.pheromone):
            for j, col in enumerate(self.__graph.pheromone):
                self.__graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    self.__graph.pheromone[i][j] += ant.get_pheromone_delta()[i][j]

    def solve(self):
        best_cost = float('inf')
        best_solution = []
        for generation in range(self.generations):
            ants = [Ant(self, self.__graph) for _ in range(self.ant_count)]
            for ant in ants:
                for _ in range(self.__graph.rank - 1):
                    ant.select_next()
                ant.total_cost += self.__graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.tabu
                ant.update_pheromone_delta()
            if uniform(0, 1) >= self.__dinamic:
                print('Schimbare.')
                best_cost = self.__change_node(best_solution)
            self.__update_pheromone(ants)
            print(generation, best_cost)
        return best_solution, best_cost

    def __change_node(self, best_path):
        node = randint(0, self.__graph.rank - 1)
        x = float(randint(self.__lowest[0], self.__highest[0]))
        y = float(randint(self.__lowest[1], self.__highest[1]))
        while [x, y] in self.__graph.network['points']:
            x = float(randint(self.__lowest[0], self.__highest[0]))
            y = float(randint(self.__lowest[1], self.__highest[1]))
        self.__graph.network['points'][node] = [x, y]
        for index in range(self.__graph.rank):
            x1, y1 = self.__graph.network['points'][index]
            self.__graph.matrix[node][index] = float((x - x1) ** 2 + (y - y1) ** 2)
        new_path = 0
        for x, y in zip(best_path[:-1], best_path[1:]):
            new_path += self.__graph.matrix[x][y]
        return new_path

    def __find_lowest_highest(self):
        maxim = [0, 0]
        minim = [float('inf'), float('inf')]
        for x, y in self.__graph.network['points']:
            if x + y < sum(minim):
                minim = [int(x), int(y)]
            elif x + y > sum(maxim):
                maxim = [int(x), int(y)]
        return minim, maxim
