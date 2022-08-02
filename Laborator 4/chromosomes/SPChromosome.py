from random import randint


class SPChromosome:

    def __init__(self, problem_parameters=None):
        self.__problem_parameters = problem_parameters
        self.__representation = []
        self.__create_representation()
        self.__fitness = 0.0

    def __create_path(self, source, destination):
        path = [source]
        current_node = source
        while current_node != destination:
            current_node = self.__choose_neighbour(path, current_node)
            path.append(current_node)
        return path[:-1]

    def __choose_neighbour(self, path, node):
        neighbours = self.__problem_parameters['matrix'][node]
        position = randint(0, len(neighbours) - 1)
        while position in path:
            position = randint(0, len(neighbours) - 1)
        return position

    def __create_representation(self):
        source, destination = self.__problem_parameters['source_and_destination']
        self.__representation = self.__create_path(source, destination) + [destination]

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
        if len(self.__representation) == 2 or len(parent.representation):
            new = self if randint(0, 100) / 100 < 0.5 else parent
            child = SPChromosome(self.__problem_parameters)
            child.representation = new.__representation
            return child
        positions = []
        minim = len(self.__representation) if len(self.__representation) < len(parent.representation) else len(
            parent.representation)
        for index in range(1, minim - 1):
            if self.__representation[index] == parent.representation[index]:
                positions.append(index)
        position = positions[randint(0, len(positions))]
        child_representation = self.__representation[:position]
        for index in range(position, len(parent.representation)):
            if parent.representation[index] in child_representation:
                continue
            child_representation.append(parent.representation[index])
        child = SPChromosome(self.__problem_parameters)
        child.representation = child_representation
        return child

    def mutation(self):
        # position that is not source or destination
        if len(self.__representation) == 2:
            self.__representation = self.__create_path(self.__representation[0], self.__representation[1]) + [self.__representation[1]]
            return
        position = randint(1, len(self.__representation) - 2)
        source = self.__representation[position - 1]
        destination = self.__representation[position + 1]
        used = self.__representation[:position - 1] + self.__representation[position + 2:]
        new_path = [source]
        while source != destination:
            source = self.__choose_neighbour(new_path + used, source)
            new_path.append(source)
        self.__representation = self.__representation[:position - 1] + new_path + self.__representation[position + 2:]

    def __str__(self):
        return 'Chromosome: ' + str([i + 1 for i in self.__representation]) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return other is not None and self.__representation == other.__representation and self.__fitness == other.__fitness
