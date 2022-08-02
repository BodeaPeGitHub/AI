from random import randint
import random


class TSPChromosome:

    def __init__(self, problem_parameters=None):
        self.__problem_parameters = problem_parameters
        self.__representation = []
        self.__create_representation()
        self.__fitness = 0.0

    def __create_representation(self):
        number_of_nodes = self.__problem_parameters['noNodes']
        self.__representation = [None] * (number_of_nodes + 1)
        for node in range(number_of_nodes):
            position = randint(0, len(self.__representation) - 2)
            while self.__representation[position] is not None:
                position = randint(0, len(self.__representation) - 2)
            self.__representation[position] = node
        self.__representation[-1] = self.__representation[0]


    @property
    def representation(self):
        return self.__representation

    @property
    def fitness(self):
        return self.__fitness

    @representation.setter
    def representation(self, l=[]):
        self.__representation = l

    @fitness.setter
    def fitness(self, value=0.0):
        self.__fitness = value

    def crossover(self, parent):
        middle = randint(0, len(self.__representation) - 1)
        child_representation = self.__representation[:middle] + [None] * (len(self.__representation) - middle)
        parent = parent.representation
        position = 0
        for index in range(middle, len(child_representation) - 1):
            while parent[position] in child_representation:
                position += 1
            child_representation[index] = parent[position]
        child_representation[-1] = child_representation[0]
        children = TSPChromosome(self.__problem_parameters)
        children.representation = child_representation
        return children

    def mutation(self):
        first_position = randint(0, len(self.__representation) - 1)
        second_position = randint(0, len(self.__representation) - 1)
        self.__representation[first_position], self.__representation[second_position] = self.__representation[
                                                                                            second_position], \
                                                                                        self.__representation[
                                                                                            first_position]
        if first_position in [0, len(self.__representation) - 1]:
            self.__representation[-1] = self.__representation[0] = self.__representation[first_position]
        if second_position in [0, len(self.__representation) - 1]:
            self.__representation[-1] = self.__representation[0] = self.__representation[second_position]


    def __str__(self):
        return 'Chromosome: ' + str([i + 1 for i in self.__representation]) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return other is not None and self.__representation == other.__representation and self.__fitness == other.__fitness
