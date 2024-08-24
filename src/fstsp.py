import elkai
import gurobipy as gp
from gurobipy import GRB
import math
import networkx as nx
import numpy as np
from baseline import Baseline
from utils import mst_partition


class MultiAgentFlyingSidekickTSP(Baseline):
    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6, theta=(0.5, 0.5)):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.groups = {depot: [] for depot in depots}
        self.solution = []
        self.cost = 0
        self.theta = theta
        self.const = math.sqrt(2)
        self.regions = {**{city: [node for node in self.graph.nodes if
                                  self.distance['drone'][node][city] < self.limit / 2] for city in cities},
                        **{depot: [depot] for depot in depots}}

    def set_nn(self, theta):
        for city in self.cities:
            distance = float('inf')
            depot = None
            for _depot in self.depots:
                _distance = min([self.distance['truck'][_depot][mid] + self.distance['drone'][mid][city] / self.speed
                                 for mid in self.graph.nodes if self.distance['drone'][mid][city] <= self.limit * theta])
                if _distance < distance:
                    distance = _distance
                    depot = _depot
            self.groups[depot].append(city)

    def set_mst(self, convex_sets):
        # construct the fully connected graph between convex sets and depots
        graph = nx.Graph()
        for depot in self.depots:
            graph.add_node(depot)
        for city in self.cities:
            graph.add_node(city)
        # distance computed as Eq. (3) in the paper
        for depot in self.depots:
            for city in self.cities:
                weight = self.distance['truck'][depot][city]
                for node in convex_sets[city]:
                    weight = min(weight, self.distance['truck'][depot][node] +
                                 self.distance['drone'][node][city] / self.speed * self.const)
                graph.add_edge(depot, city, weight=weight)
            for _depot in self.depots:
                graph.add_edge(depot, _depot, weight=self.distance['truck'][depot][_depot])
        for city in self.cities:
            for _city in self.cities:
                weight = self.distance['truck'][city][_city]
                for node in convex_sets[city]:
                    for _node in convex_sets[_city]:
                        weight = min(weight, self.distance['truck'][node][_node] +
                                     self.distance['drone'][city][node] / self.speed * self.const
                                     + self.distance['drone'][_city][_node] / self.speed * self.const)
                graph.add_edge(city, _city, weight=weight)
        self.groups = mst_partition(graph, self.depots, self.cities)

    @staticmethod
    def cut_off(x, y):
        return x if x <= y else 100000

    def lkh(self, depot, cities):  # solve tsp via LKH
        if len(cities) == 0:
            return [0, 0]
        elif len(cities) == 1:
            return [0, 1, 0]
        else:
            nodes = [depot] + cities
            int_matrix = [[self.distance['truck'][start][end] for end in nodes] for start in nodes]
            route = elkai.DistanceMatrix(int_matrix).solve_tsp()
            return route

    @staticmethod
    def set_tsp(convex_sets, distance, convex_set_distance):
        n = len(convex_sets)
        model = gp.Model('Set-TSP')
        model.setParam("OutputFlag", 0)
        # first write a tsp for the visiting order of convex sets using GG model
        select = model.addMVar((n, n), vtype=GRB.BINARY)
        model.addConstrs(select[u, u] == 0 for u in range(n))
        model.addConstrs(np.ones((n,)) @ select[:, v] == 1 for v in range(n))
        model.addConstrs(np.ones((n,)) @ select[u, :] == 1 for u in range(n))
        flow = model.addMVar((n, n), vtype=GRB.CONTINUOUS)
        model.addConstrs(flow[u, v] <= n * select[u, v] for u in range(n) for v in range(n))
        model.addConstr(np.ones((n,)) @ flow[0, :] == n - 1)
        model.addConstr(np.ones((n,)) @ flow[:, 0] == 0)
        model.addConstrs(flow[u, u] == 0 for u in range(n))
        model.addConstrs(np.ones((n,)) @ flow[:, v] - np.ones((n,)) @ flow[v, :] == 1 for v in range(1, n))

        # internal is the selection of node pair inside each convex set
        internal = [[[model.addVar(vtype=GRB.BINARY) for _ in convex_set] for _ in convex_set] for convex_set in
                    convex_sets]
        # external is the selection of node pair between two convex sets
        external = [[[[model.addVar(vtype=GRB.BINARY) for _ in v] for _ in u] for v in convex_sets] for u in
                    convex_sets]
        model.addConstrs(gp.quicksum([internal[i][j][k] for j in range(len(convex_sets[i]))
                                      for k in range(len(convex_sets[i]))]) == 1 for i in range(n))
        model.addConstrs(gp.quicksum([external[u][v][i][j] for i in range(len(convex_sets[u]))
                                      for j in range(len(convex_sets[v]))]) == select[u, v]
                         for u in range(n) for v in range(n))
        # node j in convex sets v should have same out degree internal and in degree external
        model.addConstrs(gp.quicksum([external[u][v][i][j] for u in range(n) for i in range(len(convex_sets[u]))]) ==
                         gp.quicksum([internal[v][j][k] for k in range(len(convex_sets[v]))]) for v in range(n)
                         for j in range(len(convex_sets[v])))
        # node i in convex sets u should have same in degree internal and out degree external
        model.addConstrs(gp.quicksum([external[u][v][i][j] for v in range(n) for j in range(len(convex_sets[v]))]) ==
                         gp.quicksum([internal[u][k][i] for k in range(len(convex_sets[u]))]) for u in range(n)
                         for i in range(len(convex_sets[u])))
        model.setObjective(gp.quicksum([convex_set_distance[i][j][k] * internal[i][j][k] for i in range(n)
                                        for j in range(len(convex_sets[i])) for k in range(len(convex_sets[i]))]) +
                           gp.quicksum([distance[u][v][i][j] * external[u][v][i][j] for u in range(n) for v in range(n)
                                        for i in range(len(convex_sets[u])) for j in range(len(convex_sets[v]))]),
                           GRB.MINIMIZE)
        model.optimize()

        seq = [0]
        while seq.count(0) < 2:
            for j in range(n):
                if select[seq[-1], j].X > 0.99:
                    seq.append(j)
                    break
        return seq

    def local_search_multi_drone_appr(self, seq, depot):
        # value as defined in Eq. (13)
        value = [{node: float('inf') for node in self.graph.nodes} for _ in range(2 * len(seq) - 2)]
        value[0][depot] = 0
        # appr is the time defined in Eq. (9)
        appr = [[{node: {} for node in self.regions[city]} for _ in range(self.drone)] for city in seq[1:-1]]
        # tour record the optimal tour to reach each time (appr value) defined in Eq. (9)
        tour = [[{node: {} for node in self.regions[city]} for _ in range(self.drone)] for city in seq[1:-1]]
        # group record the number of customers before visited together with the current customer
        group = [1 for _ in seq[1:-1]]  # seq[i] has the same departure node as seq[i - group[i - 1] + 1]
        # prefix record the optimal tour to reach the corresponding value in Eq. (13)
        prefix = [{node: {'truck': [], 'drone': []} for node in self.graph.nodes} for _ in range(2 * len(seq) - 2)]
        prefix[0][depot]['truck'] = [depot]

        for i in range(1, len(seq) - 1):
            for node in self.regions[seq[i]]:
                for _node in self.regions[seq[i]]:
                    # initialize appr as Eq. (8)
                    drone_time = self.distance['drone'][seq[i]][_node] + self.distance['drone'][seq[i]][node]
                    truck_time = self.distance['truck'][_node][seq[i]] + self.distance['truck'][seq[i]][node]

                    tour[i - 1][0][node][_node] = {'truck': [], 'drone': []}
                    if drone_time <= self.limit:
                        appr[i - 1][0][node][_node] = max(drone_time / self.speed, self.distance['truck'][_node][node])
                        tour[i - 1][0][node][_node]['truck'] = [_node, node]
                        tour[i - 1][0][node][_node]['drone'] = [[_node, seq[i], node]]
                    else:
                        appr[i - 1][0][node][_node] = truck_time
                        tour[i - 1][0][node][_node]['truck'] = [_node, seq[i], node]

            # track the number of possible customers to be visited together
            while i - 1 - group[i - 1] >= 0 and group[i - 1] <= group[i - 2] and group[i - 1] < self.drone:
                if self.distance['drone'][seq[i - group[i - 1]]][seq[i]] < 2 * self.limit:
                    group[i - 1] += 1
                else:
                    break
            for j in range(1, group[i - 1]):
                # assume throw all together, then
                for node in self.regions[seq[i]]:
                    for _node in self.regions[seq[i - j]]:
                        # Eq. (9)
                        appr[i - 1][j][node][_node] = float('inf')
                        tour[i - 1][j][node][_node] = {'truck': [], 'drone': []}

                        consumption = self.distance['drone'][seq[i]][_node] + self.distance['drone'][seq[i]][node]
                        if consumption > self.limit:
                            continue
                        for _mid in self.regions[seq[i - 1]]:
                            cost = max(appr[i - 2][j - 1][_mid][_node] + self.distance['truck'][_mid][node],
                                       consumption / self.speed)
                            if cost < appr[i - 1][j][node][_node]:
                                appr[i - 1][j][node][_node] = cost
                                tour[i - 1][j][node][_node]['truck'] = tour[i - 2][j - 1][_mid][_node][
                                                                           'truck'].copy() + [node]
                                tour[i - 1][j][node][_node]['drone'] = tour[i - 2][j - 1][_mid][_node][
                                                                           'drone'].copy() + [[_node, seq[i], node]]

        for i in range(1, len(seq) - 1):
            for node in self.regions[seq[i]]:
                for _node in self.regions[seq[i - 1]]:
                    # initialize value as Eq. (12)
                    if value[2 * i - 2][_node] + self.distance['truck'][_node][node] < value[2 * i - 1][node]:
                        value[2 * i - 1][node] = value[2 * i - 2][_node] + self.distance['truck'][_node][node]
                        prefix[2 * i - 1][node]['truck'] = prefix[2 * i - 2][_node]['truck'].copy() + [node]
                        prefix[2 * i - 1][node]['drone'] = prefix[2 * i - 2][_node]['drone'].copy()

            for j in range(min(self.drone, i)):
                for node in self.regions[seq[i]]:
                    for _node in self.regions[seq[i - j]]:
                        # approximation method to estimate the time consumption as Eq. (13)
                        if _node not in appr[i - 1][j][node].keys():
                            continue
                        if value[2 * i - 2 * j - 1][_node] + appr[i - 1][j][node][_node] < value[2 * i][node]:
                            value[2 * i][node] = value[2 * i - 2 * j - 1][_node] + appr[i - 1][j][node][_node]
                            prefix[2 * i][node]['truck'] = prefix[2 * i - 2 * j - 1][_node]['truck'].copy() + \
                                                           tour[i - 1][j][node][_node]['truck'].copy()
                            prefix[2 * i][node]['drone'] = prefix[2 * i - 2 * j - 1][_node]['drone'].copy() + \
                                                           tour[i - 1][j][node][_node]['drone'].copy()

        for node in self.regions[seq[-2]]:
            # after visiting the last customer, return to the depot
            if value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot] < value[2 * len(seq) - 3][depot]:
                value[2 * len(seq) - 3][depot] = value[2 * len(seq) - 4][node] + self.distance['truck'][node][depot]
                prefix[2 * len(seq) - 3][depot]['truck'] = prefix[2 * len(seq) - 4][node]['truck'].copy() + [depot]
                prefix[2 * len(seq) - 3][depot]['drone'] = prefix[2 * len(seq) - 4][node]['drone'].copy()

        return prefix[2 * len(seq) - 3][depot], value[2 * len(seq) - 3][depot]

    def get_seq(self, depot, convex_sets):
        if self.theta[1] == 0:
            seq = self.lkh(depot, self.groups[depot])
        else:
            set_distance = [[[max(self.distance['truck'][k][j],
                                  self.cut_off((self.distance['drone'][j][city] + self.distance['drone'][city][k]),
                                               self.limit)) / self.speed for j in convex_set] for k in convex_set]
                            for convex_set, city in zip(convex_sets, [depot] + self.groups[depot])]
            distance = [[[[self.distance['truck'][i][j] for j in v] for i in u]
                         for v in convex_sets] for u in convex_sets]
            seq = self.set_tsp(convex_sets, distance, set_distance)
        return seq

    def single_solution(self, depot, convex_sets):
        cities = self.groups[depot]
        if len(cities) == 0:
            return {'truck': [depot, depot], 'drone': []}

        seq = self.get_seq(depot, convex_sets)
        seq = [depot] + [cities[i - 1] for i in seq[1:-1]] + [depot]
        solution, cost = self.local_search_multi_drone_appr(seq, depot)
        self.cost += cost
        return solution

    def convert(self, solution):
        route = {'truck': solution['truck'], 'drone': [[] for _ in range(self.drone)]}
        route['drone'][0] = solution['drone']
        return route

    # not necessary a convex set, a wrong name
    def get_convex_sets(self, theta):
        convex_sets = {city: [] for city in self.cities}
        for node in self.graph.nodes:
            closest_city = None
            closest_distance = self.limit * theta
            for city in self.cities:
                if self.distance['drone'][node][city] <= closest_distance:
                    closest_city = city
                    closest_distance = self.distance['drone'][node][city]
            if closest_city is not None:
                convex_sets[closest_city].append(node)
        return convex_sets

    def get_boundary_convex_sets(self, theta):
        convex_sets = self.get_convex_sets(theta)
        boundary_convex_sets = {city: [] for city in self.cities}
        for city in self.cities:
            for node in convex_sets[city]:
                for neighbor in nx.neighbors(self.graph, node):
                    if neighbor not in convex_sets[city]:
                        boundary_convex_sets[city].append(node)
                        break
        return boundary_convex_sets

    def solve(self):
        convex_sets = self.get_boundary_convex_sets(self.theta[0])
        self.set_mst(convex_sets)
        for depot in self.depots:
            convex_set = [[depot]] + [convex_sets[city] for city in self.groups[depot]]
            solution = self.single_solution(depot, convex_set)
            self.solution.append(self.convert(solution))
        return self.solution, self.cost

    # function that helps fast generate the results for the ablation study of the drone numbers
    def solve_multiple_drones(self):
        costs = []
        convex_sets = self.get_boundary_convex_sets(self.theta[0])
        self.set_mst(convex_sets)
        for depot in self.depots:
            convex_set = [[depot]] + [convex_sets[city] for city in self.groups[depot]]
            cities = self.groups[depot]
            if len(cities) > 0:
                seq = self.get_seq(depot, convex_set)
                seq = [depot] + [cities[i - 1] for i in seq[1:-1]] + [depot]
                for drone in range(6):
                    self.drone = drone
                    _, cost = self.local_search_multi_drone_appr(seq, depot)
                    costs.append(cost)
        return costs
