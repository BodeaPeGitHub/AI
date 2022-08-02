from random import randint


class Chromosome:

    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__representation = []
        self.__communities_dictionary = {}
        self.__create_representation()
        self.__fitness = 0.0

    def __create_representation(self):
        commnunites_number = self.__problParam['noCommunities']
        self.__representation = [0] * self.__problParam['noNodes']
        while commnunites_number != 0:
            self.__communities_dictionary[commnunites_number] = 1
            pos = randint(0, len(self.__representation) - 1)
            while self.__representation[pos] != 0:
                pos = randint(0, len(self.__representation) - 1)
            self.__representation[pos] = commnunites_number
            commnunites_number -= 1
        for i in range(len(self.__representation)):
            if self.__representation[i] == 0:
                community = randint(1, self.__problParam['noCommunities'])
                self.__representation[i] = community
                self.__communities_dictionary[community] += 1

    @property
    def representation(self):
        return self.__representation

    @property
    def fitness(self):
        return self.__fitness

    @property
    def communities_dictionary(self):
        return self.__communities_dictionary

    @representation.setter
    def representation(self, l=[]):
        self.__representation = l

    @fitness.setter
    def fitness(self, value=0.0):
        self.__fitness = value

    @communities_dictionary.setter
    def communities_dictionary(self, value={}):
        self.__communities_dictionary = value

    def crossover(self, c):
        r = randint(0, len(self.__representation) - 1)
        new_representation = []
        for i in range(r):
            new_representation.append(self.__representation[i])
        for i in range(r, len(self.__representation)):
            new_representation.append(c.__representation[i])

        new_dictionary = {}
        for elem in new_representation:
            if elem in new_dictionary:
                new_dictionary[elem] += 1
            else:
                new_dictionary[elem] = 1

        for elem in range(1, self.__problParam['noCommunities'] + 1):
            while elem not in new_dictionary:
                pos = randint(0, len(new_representation) - 1)
                if new_dictionary[new_representation[pos]] > 1:
                    new_dictionary[elem] = 1
                    new_representation[pos] = elem

        offspring = Chromosome(c.__problParam)
        offspring.representation = new_representation
        offspring.communities_dictionary = new_dictionary
        return offspring

    def mutation(self):
        pos = randint(0, len(self.__representation) - 1)
        while self.__communities_dictionary[self.__representation[pos]] <= 1:
            pos = randint(0, len(self.__representation) - 1)
        self.__communities_dictionary[self.__representation[pos]] -= 1
        community = randint(1, self.__problParam['noCommunities'])
        self.__communities_dictionary[community] += 1
        self.__representation[pos] = community

    def __str__(self):
        return '\nChromo: ' + str(self.__representation) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__representation == c.__representation and self.__fitness == c.__fitness