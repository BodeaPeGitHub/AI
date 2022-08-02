from random import randint
from chromosomes.TSPChromosome import TSPChromosome


class GA:
    def __init__(self, parameters=None, problem_parameters=None, chromosome=None):
        self.__parameters = parameters
        self.__chromosome = chromosome
        self.__problem_parameters = problem_parameters
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(self.__parameters['popSize']):
            self.__population.append(self.__chromosome(self.__problem_parameters))

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problem_parameters['function'](c.representation, self.__problem_parameters)

    def best_chromosome(self):
        best = self.__population[0]
        for chromosome in self.__population:
            if chromosome.fitness < best.fitness:
                best = chromosome
        return best

    def find_all_best_chromosomes(self):
        best = [self.__population[0]]
        for chromosome in self.__population:
            if chromosome.fitness == best[0].fitness and chromosome not in best:
                best.append(chromosome)
            if chromosome.fitness < best[0].fitness:
                best = [chromosome]
        return best

    def worst_chromosome(self):
        worst = self.__population[0]
        for chromosome in self.__population:
            if chromosome.fitness > worst.fitness:
                worst = chromosome
        return worst

    def selection(self):
        first_position = randint(0, self.__parameters['popSize'] - 1)
        second_position = randint(0, self.__parameters['popSize'] - 1)
        if self.__population[first_position].fitness < self.__population[second_position].fitness:
            return first_position
        return second_position

    def one_generation(self):
        new_population = []
        for _ in range(self.__parameters['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            new_population.append(off)
        self.__population = new_population
        self.evaluation()

    def one_generation_elitism(self):
        new_population = [self.best_chromosome()]
        for _ in range(self.__parameters['popSize'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            new_population.append(off)
        self.__population = new_population
        self.evaluation()

    def __find_worst_index(self):
        worst_fitness = self.__population[0].fitness
        index = 0
        for i in range(len(self.__population)):
            if worst_fitness < self.__population[i].fitness:
                worst_fitness = self.__population[i].fitness
                index = i
        return index

    def one_generation_steady_state(self):
        for _ in range(self.__parameters['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            off.fitness = self.__problem_parameters['function'](off.representation, self.__problem_parameters)
            worst_index = self.__find_worst_index()
            if off.fitness < self.__population[worst_index].fitness:
                self.__population[worst_index] = off
