# the file defines the template of the baseline class
class Baseline:
    def __init__(self, graph, depots, cities, distance, drone, limit, speed):
        self.graph = graph
        self.depots = depots
        self.cities = cities
        self.distance = distance
        self.name = name
        self.drone = drone
        self.limit = limit
        self.speed = speed

    def convert(self, solution):
        # all the solution should be in the format
        # {'truck': [node1, node2, ...], 'drone': [[[node1, city, node2], ...], ...]}
        # not necessarily to implement this method
        pass

    def solve(self):
        raise NotImplementedError
