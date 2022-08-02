class Graph:
    def __init__(self, network: dict):
        self.network = network
        self.matrix = network['matrix']
        self.rank = len(self.matrix)
        self.pheromone = [[1 / (self.rank * self.rank)] * self.rank] * self.rank
