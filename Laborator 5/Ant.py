from random import randint, random

import ACO
import Graph


class Ant:
    def __init__(self, aco: ACO, graph: Graph):
        self.__colony = aco
        self.__graph = graph
        self.total_cost = 0.0
        self.__pheromone_delta = []
        self.__allowed = [i for i in range(graph.rank)]
        self.__eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.rank)] for i in range(graph.rank)]
        self.__current = randint(0, graph.rank - 1)
        self.tabu = [self.__current]
        self.__allowed.remove(self.__current)

    def get_pheromone_delta(self):
        return self.__pheromone_delta

    def select_next(self):
        denominator = 0
        for i in self.__allowed:
            denominator += self.__graph.pheromone[self.__current][i] ** self.__colony.alpha * \
                           self.__eta[self.__current][i] ** self.__colony.beta
        probabilities = [0] * self.__graph.rank
        for i in range(self.__graph.rank):
            if i not in self.__allowed:
                continue
            probabilities[i] = self.__graph.pheromone[self.__current][i] * self.__colony.alpha * self.__eta[self.__current][i] ** self.__colony.beta / denominator
        selected = 0
        rand = random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        self.__allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.__graph.matrix[self.__current][selected]
        self.__current = selected

    def update_pheromone_delta(self):
        self.__pheromone_delta = [[0] * self.__graph.rank] * self.__graph.rank
        for i, j in zip(self.tabu[:-1], self.tabu[1:]):
            self.__pheromone_delta[i][j] = self.__colony.Q / self.total_cost
