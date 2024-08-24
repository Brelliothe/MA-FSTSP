import networkx as nx
import numpy as np
import random
from baseline import Baseline
from utils import haversine


class HillClimbingVariableNeighborhoodSearch(Baseline):
    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6, rounds=1000):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.groups = {depot: [] for depot in self.depots}
        self.rounds = rounds
        self.solution = []
        self.cost = 0
        self.neighbors = {city: [node for node in self.graph.nodes
                                 if haversine(self.graph.nodes[city]['pos'], self.graph.nodes[node]['pos']) < limit]
                          for city in np.concatenate((depots, cities))}

    def partition(self):
        # nearest neighbor assign groups
        for city in self.cities:
            distance = float('inf')
            closest_depot = None
            for depot in self.depots:
                path_length = nx.shortest_path_length(self.graph, source=depot, target=city, weight='weight')
                if path_length < distance:
                    distance = path_length
                    closest_depot = depot
            self.groups[closest_depot].append(city)

    def init_solution(self, nodes):
        # use heuristic tsp to get initial solution
        if len(nodes) == 1:
            return [(nodes[0], -1, nodes[0])]
        elif len(nodes) == 2:
            return [(nodes[0], -1, nodes[0]), (nodes[1], -1, nodes[1]), (nodes[0], -1, nodes[0])]
        graph = nx.Graph()
        for i in nodes:
            for j in nodes:
                graph.add_edge(i, j, weight=nx.dijkstra_path_length(self.graph, i, j, weight='weight'))
        # nx.tsp produces a tsp loop, not a path
        solution = nx.approximation.traveling_salesman_problem(graph, weight='weight', nodes=nodes,
                                                               method=nx.approximation.greedy_tsp)
        # initially all customers visited by truck (index = -1)
        solution = [(i, -1, i) for i in solution]
        return solution

    def neighborhood_search(self, solution):
        # randomly choose a neighborhood search method, 4 means do not search
        num = random.randint(1, 5)
        cost = self.solution_cost(solution)
        if num == 1:  # change a drone's location
            free_nodes = [i for i in range(len(solution)) if solution[i][1] > -1]
            if len(free_nodes) == 0:
                return solution
            index = random.choice(free_nodes)
            _solution = solution.copy()
            for node in self.graph.neighbors(solution[index][0]):
                _solution[index] = (node, solution[index][1], solution[index][2])
                _cost = self.solution_cost(_solution)
                if _cost < cost:
                    solution = _solution
                    cost = _cost
        if num == 2:  # replace a truck visitor by a drone
            truck_nodes = [i for i in range(1, len(solution) - 1) if solution[i][1] == -1]
            if len(truck_nodes) == 0:
                return solution
            # randomly choose 1 truck node
            index = random.choice(truck_nodes)
            # check if there is a free drone
            drones_on_sky = [0 for _ in range(self.drone)]
            for i in range(index):
                _, d, city = solution[i]
                if d > -1:
                    drones_on_sky[d] += 1 if city > 0 else -1
            free_drones = [i for i in range(self.drone) if drones_on_sky[i] == 0]
            if len(free_drones) == 0:
                return solution
            # arbitrarily choose a drone on truck
            drone = random.choice(free_drones)
            city = solution[index][2]
            # find the best take-off and land nodes
            best_pair = None
            for start in self.neighbors[city]:
                for end in self.neighbors[city]:
                    _solution = solution[:index].copy() + [(start, drone, city), (end, drone, -city)] + \
                                solution[index + 1:].copy()
                    _cost = self.solution_cost(_solution)
                    if _cost < cost:
                        cost = _cost
                        best_pair = (start, end)
            if best_pair is not None:
                solution = solution[:index].copy() + [(best_pair[0], drone, city), (best_pair[1], drone, -city)] + \
                           solution[index + 1:].copy()
        if num == 3:  # change the visiting order of nodes
            best_pair = None
            for i in range(1, len(solution) - 2):
                _solution = solution[:i].copy() + [solution[i + 1]] + [solution[i]] + solution[i + 2:].copy()
                _cost = self.solution_cost(_solution)
                if _cost < cost:
                    cost = _cost
                    best_pair = (i, i + 1)
            if best_pair is not None:
                solution = solution[:best_pair[0]].copy() + [solution[best_pair[1]]] + [solution[best_pair[0]]] + \
                           solution[best_pair[1] + 1:].copy()
        return solution

    def solution_cost(self, solution):
        # solution is formatted as a list of (key node, index, city) tuples, key node is the node name in graph,
        # index is the tuple's visitor, i.e., -1 -> truck, 0 ~ n - 1 -> drone, city is the where drone/truck visit
        cost = [0 for _ in solution]
        for i in range(1, len(solution)):
            cost[i] = cost[i - 1] + self.distance['truck'][solution[i - 1][0]][solution[i][0]]
            if solution[i][1] > -1 and solution[i][2] < 0:
                for j in range(1, i):
                    if solution[j][-1] + solution[i][-1] == 0:
                        drone_distance = self.distance['drone'][solution[j][0]][solution[j][-1]] + \
                                         self.distance['drone'][solution[j][-1]][solution[i][0]]
                        drone_distance = 1000000 if drone_distance > self.limit else drone_distance
                        cost[i] = max(cost[i], cost[j] + drone_distance / self.speed)
        return cost[-1]

    def convert(self, solution):
        # convert the solution to a uniform format
        route = {'truck': [node for node, _, _ in solution], 'drone': [[] for d in range(self.drone)]}
        for i in range(1, len(solution) - 1):
            for j in range(i + 1, len(solution) - 1):
                if solution[i][-1] + solution[j][-1] == 0:
                    city = max(solution[i][-1], solution[j][-1])
                    route['drone'][solution[i][1]].append([solution[i][0], city, solution[j][0]])
        return route

    def single_solution(self, depot):
        # for single truck problem, do the neighborhood search
        solution = self.init_solution([depot] + self.groups[depot])
        for _ in range(self.rounds):
            solution = self.neighborhood_search(solution)
        self.cost += self.solution_cost(solution)
        _solution = self.convert(solution)
        self.solution.append(_solution.copy())

    def solve(self):
        self.partition()
        for depot in self.depots:
            self.single_solution(depot)
        return self.solution, self.cost
