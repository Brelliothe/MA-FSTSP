import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
from baseline import Baseline
from utils import nearest_node_except_self, asymmetric_traveling_salesman_problem


class LinearRelaxedMasterProblem(Baseline):
    # From 'Scheduling trucks and drones for cooperative deliveries'
    # Implement the VNS with acceleration strategy + strategy 2 to obtain the solution
    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.routes = []
        # binary variable, alpha[i][j][k] = 1 iff truck k travels from node i to node j
        self.alpha = [[[0 for _ in depots] for _ in graph.nodes] for _ in graph.nodes]
        # binary variable, beta[i][j][k][d] = 1 iff drone d on truck k flies from i to j
        self.beta = [[[[0 for _ in range(drone)] for _ in depots] for _ in graph.nodes] for _ in graph.nodes]
        # binary variable, epsilon[k] = 1 iff truck k is used
        self.epsilon = [0 for _ in depots]
        self.nodes = list(self.graph.nodes)
        self.city_indices = {city: self.nodes.index(city) for city in self.cities}
        self.depot_indices = {depot: self.nodes.index(depot) for depot in self.depots}
        self.m = 1000000  # a very large number for the big M method
        self.y = [[[[] for _ in range(self.drone)] for _ in graph.nodes] for _ in graph.nodes]
        self.solution = []
        self.cost = 0

    def update_y(self):
        # call it each time appending a new route to update y
        route = self.routes[-1]['drone']
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                for d in range(self.drone):
                    self.y[i][j][d].append(0)
        for d in range(self.drone):
            route_d = route[d]
            for path in route_d:
                start, city, end = path[0], path[1], path[2]
                self.y[self.nodes.index(start)][self.nodes.index(city)][d][-1] = 1

    def initial_solution(self):
        # assign each city to truck groups
        groups = [[depot] for depot in self.depots]
        for city in self.cities:
            index = 0
            for i in range(len(self.depots)):
                if self.distance['truck'][self.depots[i]][city] < self.distance['truck'][self.depots[index]][city]:
                    index = i
            groups[index].append(city)
        # for each truck group, initialize the routes
        for i in range(len(groups)):
            group = groups[i].copy()
            if len(group) == 1:
                continue
            matching = {city: nearest_node_except_self(self.graph, city) for city in group[1:]}
            nodes_must_visit = group[:1] + list(matching.values())
            # use greedy algorithm to determine the truck route
            if len(nodes_must_visit) > 2:
                solution = asymmetric_traveling_salesman_problem(self.graph, nodes_must_visit)
            elif len(nodes_must_visit) == 2:
                solution = nx.dijkstra_path(self.graph, nodes_must_visit[0], nodes_must_visit[1]) + \
                           nx.dijkstra_path(self.graph, nodes_must_visit[1], nodes_must_visit[0])[1:]
            else:
                solution = [nodes_must_visit[0], nodes_must_visit[0]]
            if solution[0] != nodes_must_visit[0]:
                index = solution.index(nodes_must_visit[0])
                solution = solution[index:] + solution[1: index + 1]
            # compute the shortest path connecting each node
            route = {'truck': solution, 'drone': [[] for _ in range(self.drone)], 'cost': 0,
                     'cities': group[1:].copy()}
            visited = group[1:].copy()
            for start, end in zip(solution[:-1], solution[1:]):
                for city in group[1:]:
                    if end == matching[city] and city in visited:
                        visited.remove(city)
                        route['drone'][0].append([start, city, end])
                        route['cost'] += max(self.distance['truck'][start][end],
                                             (self.distance['drone'][start][city] +
                                             self.distance['drone'][city][end]) / self.speed)
                        route['cost'] -= self.distance['truck'][start][end]
                route['cost'] += self.distance['truck'][start][end]
            self.routes.append(route)
            self.update_y()

    def solve_truck_route(self, nodes_must_visit):
        # nodes_must_visit is a list of nodes to visit, [depot] + list of road nodes
        # duplicate the depot to separate the in and out, name it as the nodes + 1 node
        # reorder all the nodes in graph.nodes, follows by nodes_must_visit + other nodes
        # now 0 is the depot to leave and nodes is the depot to return
        _nodes = nodes_must_visit.copy()
        for node in self.graph.nodes:
            if node not in _nodes:
                _nodes.append(node)
        _nodes.append(_nodes[0])
        nodes = len(_nodes) - 1
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # binary, if the truck travels from node i to node j, alpha[i][j] = 1
        alpha = model.addMVar((nodes + 1, nodes + 1), vtype=GRB.BINARY)
        # integer, if the truck starts from the depot, node i is visited in mu[i]-th
        mu = model.addMVar((nodes + 1,), vtype=GRB.INTEGER)
        # continuous non-negative, delta[i] is the time truck group arriving at node i
        delta = model.addMVar((nodes + 1,), vtype=GRB.CONTINUOUS)

        # minimize overall visiting time, e.g. sum([time[i][j] * alpha[i][j]])
        model.setObjective(gp.quicksum([self.distance['truck'][_nodes[i]][_nodes[j]] * alpha[i, j]
                                        for i in range(nodes) for j in range(1, nodes + 1)]))
        # depot has one in degree and one out degree
        model.addConstr(np.ones((nodes + 1,)) @ alpha[0, :] == 1)
        model.addConstr(np.ones((nodes + 1,)) @ alpha[:, -1] == 1)
        model.addConstr(np.ones((nodes + 1,)) @ alpha[:, 0] == 0)
        model.addConstr(np.ones((nodes + 1,)) @ alpha[-1, :] == 0)

        # in degree equals to out degree for every node
        model.addConstrs(np.ones((nodes + 1,)) @ alpha[i, :] == np.ones((nodes + 1,)) @ alpha[:, i]
                         for i in range(1, nodes))

        # for node must visit, in == out == 1
        model.addConstrs(np.ones((nodes + 1,)) @ alpha[i, :] == 1 for i in range(1, len(nodes_must_visit)))
        model.addConstrs(np.ones((nodes + 1,)) @ alpha[:, i] == 1 for i in range(1, len(nodes_must_visit)))

        # for other nodes, in == out == 0
        model.addConstrs(np.ones((nodes + 1,)) @ alpha[i, :] == 0 for i in range(len(nodes_must_visit), nodes + 1))
        model.addConstrs(np.ones((nodes + 1,)) @ alpha[:, i] == 0 for i in range(len(nodes_must_visit), nodes))
        model.addConstr(np.ones((nodes + 1,)) @ alpha[:, 0] == 0)

        # delta[0] = 0
        model.addConstr(delta[0] == 0)

        # delta[j] >= delta[i] + truck_time[i][j] - M * (1 - alpha[i][j])
        model.addConstrs(delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                         self.m * (1 - alpha[i, j]) for i in range(nodes) for j in range(1, nodes + 1))
        # delta[j] <= delta[i] + truck_time[i][j] + M * (1 - alpha[i][j])
        model.addConstrs(delta[j] <= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] +
                         self.m * (1 - alpha[i, j]) for i in range(nodes) for j in range(1, nodes + 1))

        # mu[i] - mu[j] <= |O| * (1 - alpha[i, j]) - 1
        model.addConstrs(mu[i] - mu[j] <= (nodes + 1) * (1 - alpha[i, j]) - 1
                         for i in range(nodes) for j in range(1, nodes + 1))

        # delta[i] >= 0
        model.addConstrs(delta[i] >= 0 for i in range(nodes + 1))

        # mu[i] <= |O|
        model.addConstrs(mu[i] <= nodes + 1 for i in range(nodes + 1))

        model.optimize()
        assert model.Status == GRB.OPTIMAL, 'do not find optimal solution, check the code'

        _truck_route = sorted(nodes_must_visit[1:], key=lambda x: mu[nodes_must_visit.index(x)].X)
        _truck_route = nodes_must_visit[:1] + _truck_route + nodes_must_visit[:1]
        truck_route = nodes_must_visit[:1].copy()
        for start, end in zip(_truck_route[:-1], _truck_route[1:]):
            truck_route += nx.dijkstra_path(self.graph, start, end, weight='weight')[1:]
        return truck_route, _truck_route

    def solve_drone_route(self, truck_route, cities):
        # reorder and duplicate depot, let 0 be leaving depot and -1 be entering depot
        # duplicate all cities from 1 to len(cities), to separate city from road crossing
        for e in cities:
            assert cities.count(e) == 1, f'duplicate element in {cities}, check code'
        _nodes = [truck_route[0]] + cities
        for node in self.graph.nodes:
            if node != truck_route[0]:
                _nodes.append(node)
        _nodes.append(_nodes[0])
        nodes = len(_nodes)
        # print(_nodes[19], truck_route, len(cities), cities)

        # turn the truck_route into adjacent matrix
        # currently is directed graph
        # first compute the truck route time:
        arrival_time = [0]
        for start, end in zip(truck_route[:-1], truck_route[1:]):
            arrival_time.append(arrival_time[-1] + self.distance['truck'][start][end])

        adjacent = np.zeros((nodes, nodes))
        for start, end in zip(truck_route[:-1], truck_route[1:]):
            i = 0 if start == truck_route[0] else 1 + len(cities) + _nodes[1 + len(cities):].index(start)
            j = 1 + len(cities) + _nodes[1 + len(cities):].index(end)
            adjacent[i][j] = 1

        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # binary, if drone d move from i to j (either fly or carried by truck), beta[i][j][d] = 1
        beta = model.addMVar((nodes, nodes, self.drone), vtype=GRB.BINARY, name='edge')
        # integer, drone d visit i at phi[i][d]-th
        phi = model.addMVar((nodes, self.drone), vtype=GRB.INTEGER, name='order')
        # continuous, time for truck group at node i
        delta = model.addMVar((nodes,), vtype=GRB.CONTINUOUS, name='time')

        # minimize drone traveling time
        model.setObjective(gp.quicksum([self.distance['drone'][_nodes[i]][_nodes[c]] / self.speed * beta[i][c][d]
                                        for i in range(len(cities) + 1, nodes) for d in range(self.drone)
                                        for c in range(1, len(cities) + 1)])
                           + gp.quicksum([self.distance['drone'][_nodes[c]][_nodes[j]] / self.speed * beta[c][j][d]
                                          for d in range(self.drone) for j in range(len(cities) + 1, nodes)
                                          for c in range(1, len(cities) + 1)]))

        # start and end
        model.addConstrs((np.ones((nodes,)) @ beta[0, :, d] == 1 for d in range(self.drone)), name='leave depot')
        model.addConstrs((np.ones((nodes,)) @ beta[:, 0, d] == 0 for d in range(self.drone)), name='no enter depot')
        model.addConstrs((np.ones((nodes,)) @ beta[-1, :, d] == 0 for d in range(self.drone)), name='no leave depot')
        model.addConstrs((np.ones((nodes,)) @ beta[:, -1, d] == 1 for d in range(self.drone)), name='enter depot')

        # in equals to out except for start and end depot
        model.addConstrs((np.ones((nodes,)) @ beta[:, j, d] == np.ones((nodes,)) @ beta[j, :, d]
                          for j in range(1, nodes - 1) for d in range(self.drone)), name='degree equation')

        # cities visited
        model.addConstrs((np.ones((nodes,)) @ beta[:, c, :] @ np.ones((self.drone,)) == 1
                          for c in range(1, len(cities) + 1)), name='enter city')
        model.addConstrs((np.ones((nodes,)) @ beta[c, :, :] @ np.ones((self.drone,)) == 1
                          for c in range(1, len(cities) + 1)), name='leave city')

        # cannot visit city consecutively
        model.addConstrs((beta[i, j, d] == 0 for d in range(self.drone)
                          for i in range(1, len(cities) + 1) for j in range(1, len(cities) + 1)),
                         name='one per visit')
        # each node has at most one drone take off
        model.addConstrs((np.ones((nodes,)) @ beta[:, c, :] @ np.ones((self.drone,)) <= 1
                          for c in range(1, len(cities) + 1)), name='one take off')

        # each node has at most one drone land
        model.addConstrs((np.ones((nodes,)) @ beta[c, :, :] @ np.ones((self.drone,)) <= 1
                          for c in range(1, len(cities) + 1)), name='one land')

        model.addConstr(delta[0] == 0)
        # delta[j] >= delta[i] + time_truck[i][j] - M * (1 - alpha[i][j])
        model.addConstrs((delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                          self.m * (1 - adjacent[i][j]) for i in range(nodes - 1) for j in range(1, nodes)),
                         name='road')
        # delta[c] >= delta[i] + time_drone[i][c] - M * (1 - beta[i][c][d])
        model.addConstrs((delta[c] >= delta[i] + self.distance['drone'][_nodes[i]][_nodes[c]] / self.speed -
                          self.m * (1 - beta[i, c, d]) for i in range(nodes - 1)
                          for c in range(1, len(cities) + 1) for d in range(self.drone)), name="drone takeoff time")
        # delta[j] >= delta[c] + time_drone[j][c] - M * (1 - beta[c][j][d])
        model.addConstrs((delta[j] >= delta[c] + self.distance['drone'][_nodes[c]][_nodes[j]] / self.speed -
                          self.m * (1 - beta[c, j, d]) for j in range(1, nodes)
                          for c in range(1, len(cities) + 1) for d in range(self.drone)), name='drone land time')
        # delta[j] >= delta[i] + time_truck[i][j] - M * (1 - beta[i][j][d])
        model.addConstrs((delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                          self.m * (1 - beta[i, j, d]) for i in range(1 + len(cities), nodes - 1)
                          for j in range(1 + len(cities), nodes) for d in range(self.drone)), name='carry time')
        # delta[j] >= delta[i] + time_truck[i][j] - M * (2 - alpha[i][j] - beta[i][j][d])
        model.addConstrs((delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                          self.m * (2 - adjacent[i][j] - beta[i, j, d]) for d in range(self.drone)
                          for i in range(nodes - 1) for j in range(1, nodes)), name='stupid constraint')
        # delta[j] >= delta[i] - M * (2 - beta[i][c][d] - beta[c][j][d])
        model.addConstrs(
            (delta[j] >= delta[i] - self.m * (2 - beta[i, c, d] - beta[c, j, d]) for d in range(self.drone)
             for i in range(1 + len(cities), nodes - 1) for j in range(1 + len(cities), nodes)
             for c in range(1, len(cities) + 1)), name='visit order in time')
        # beta[i][j][d] <= alpha[i][j]
        model.addConstrs((beta[i, j, d] <= adjacent[i][j] for i in range(len(cities) + 1, nodes - 1)
                          for j in range(len(cities) + 1, nodes) for d in range(self.drone)),
                         name='carried by truck')
        # beta[i][c][d] <= sum_j([alpha[i][j]])
        model.addConstrs((beta[0, c, d] <= np.ones((nodes,)) @ adjacent[0, :] for c in range(1, len(cities) + 1)
                          for d in range(self.drone)), name='take off from depot')
        model.addConstrs(
            (beta[i, c, d] <= np.ones((nodes,)) @ adjacent[i, :] for i in range(len(cities) + 1, nodes - 1)
             for c in range(1, len(cities) + 1) for d in range(self.drone)), name='take off on truck')
        # beta[c][j][d] <= sum_i([alpha[i][j]])
        model.addConstrs((beta[c, j, d] <= np.ones((nodes,)) @ adjacent[:, j] for j in range(len(cities) + 1, nodes)
                          for c in range(1, len(cities) + 1) for d in range(self.drone)), name='land on truck')
        # phi[i][d] - phi[j][d] <= nodes * (1 - beta[i][j][d]) - 1
        model.addConstrs((phi[i, d] - phi[j, d] <= nodes * (1 - beta[i, j, d]) - 1 for i in range(nodes - 1)
                          for j in range(1, nodes) for d in range(self.drone)), name='drone order')
        # 0 <= phi[i][d] <= nodes
        model.addConstrs(phi[i, d] <= nodes for i in range(nodes) for d in range(self.drone))

        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("model.ilp")

        drone_route = [[] for _ in range(self.drone)]
        for d in range(self.drone):
            for c in range(1, len(cities)):
                for i in range(nodes):
                    if beta[i, c, d].X > 0.5:
                        for j in range(nodes):
                            if beta[c, j, d].X > 0.5:
                                drone_route[d].append([_nodes[i], _nodes[c], _nodes[j]])
        return drone_route, delta[nodes - 1].X

    def neighborhood_search(self, route, theta):
        best_cost, _route = route['cost'], route
        for node in self.graph.nodes:
            if node == route['truck'][0]:
                continue
            nodes_truck_must_visit = route['road'].copy()
            nodes_drone_must_visit = route['cities'].copy()
            if node in list(self.cities):
                if node in route['cities']:
                    nodes_drone_must_visit.remove(node)
                elif theta[node] > 0:
                    nodes_drone_must_visit.append(node)
                else:
                    continue
            else:

                if node in route['road']:
                    nodes_truck_must_visit.remove(node)
                elif node in route['truck']:
                    continue
                else:
                    nodes_truck_must_visit.append(node)
            truck_route, _truck_route = self.solve_truck_route(nodes_truck_must_visit)
            drone_route, cost = self.solve_drone_route(_truck_route, nodes_drone_must_visit)
            if cost < best_cost:
                _route = {'truck': truck_route, 'drone': drone_route, 'cost': cost,
                          'cities': nodes_drone_must_visit,
                          'road': nodes_truck_must_visit}
                best_cost = cost
        return _route, best_cost - route['cost']

    def shake(self, route):
        nodes_must_visit = [route['road'][0]] + np.random.choice(self.nodes, len(route['road']) - 1).tolist()
        truck_route, _truck_route = self.solve_truck_route(nodes_must_visit)
        drone_route, cost = self.solve_drone_route(_truck_route, route['cities'])
        if cost < route['cost']:
            self.routes.append({'truck': truck_route, 'drone': drone_route, 'cost': cost, 'cities': route['cities'],
                                'road': nodes_must_visit})
            self.update_y()

    def pricing_problem(self, route, theta):
        init_cost = route['cost']
        # according to the increasing order of theta, sort cities
        # based on sorted sequence assign cities
        cities = sorted(list(self.cities), key=lambda x: theta[x])[: len(self.cities) // len(self.depots)]
        depot = route['truck'][0]
        # solve truck route
        nodes_must_visit = [depot] + [nearest_node_except_self(self.graph, city) for city in cities]
        truck_route, _truck_route = self.solve_truck_route([*dict.fromkeys(nodes_must_visit)])
        # solve drone route
        drone_route, cost = self.solve_drone_route(_truck_route, cities)
        # based on initial solution, do neighborhood search
        route = {'truck': truck_route, 'drone': drone_route, 'cost': cost, 'cities': cities,
                 'road': nodes_must_visit}
        reduced_cost, iters = 0, 0
        while (reduced_cost < 0 and iters < 10) or iters < 1:
            route, reduced_cost = self.neighborhood_search(route, theta)
            if reduced_cost == 0:
                self.shake(route)
                iters += 1
        return route['cost'] - init_cost

    def master_problem(self):
        theta = {}
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        ksi = model.addMVar((len(self.routes),), vtype=GRB.CONTINUOUS)
        model.setObjective(gp.quicksum([ksi[i] * self.routes[i]['cost'] for i in range(len(self.routes))]))
        model.addConstr(np.ones((len(self.routes),)) @ ksi <= len(self.depots))
        for city in self.cities:
            j = self.city_indices[city]
            model.addConstr(gp.quicksum([ksi[r] * self.y[i][j][d][r] for r in range(len(self.routes))
                                         for i in range(len(self.nodes)) for d in range(self.drone)]) >= 1,
                            name=f'{j}')
        model.optimize()

        # get the dual variables of ksi
        for city in self.cities:
            j = self.city_indices[city]
            constr = model.getConstrByName(f'{j}')
            theta[city] = constr.Pi
        # get the active routes
        selected = []
        for r in range(len(self.routes)):
            if ksi[r].X > 0.5:
                selected.append(self.routes[r])
        # sort by depots order
        selected.sort(key=lambda x: list(self.depots).index(x['truck'][0]))
        return selected, theta, ksi.X

    def solve_optimal_pp(self, group):
        depot, cities = group[0], group[1:]
        _nodes = group.copy()
        for node in self.graph.nodes:
            if node != depot:
                _nodes.append(node)
        _nodes.append(depot)
        mask = np.array([0] + [1 for _ in cities] + [0 for _ in _nodes[len(group):]])
        nodes = len(_nodes)

        model = gp.Model()
        model.setParam("OutputFlag", 0)
        # binary, alpha[i][j] = 1 iff truck go from i to j
        alpha = model.addMVar((nodes, nodes), vtype=GRB.BINARY)
        # depot in and out
        model.addConstr(np.ones((nodes,)) @ alpha[0, :] == 1)
        model.addConstr(np.ones((nodes,)) @ alpha[:, -1] == 1)
        model.addConstr(np.ones((nodes,)) @ alpha[:, 0] == 0)
        model.addConstr(np.ones((nodes,)) @ alpha[-1, :] == 0)

        # in degree equals to out degree less than or equals to 1
        model.addConstrs(np.ones((nodes,)) @ alpha[i, :] == np.ones((nodes,)) @ alpha[:, i]
                         for i in range(1, nodes - 1))
        model.addConstrs(np.ones((nodes,)) @ alpha[i, :] == 0 for i in range(1, len(group)))
        model.addConstrs(np.ones((nodes,)) @ alpha[i, :] <= 1 for i in range(1, nodes - 1))

        # binary, beta[i][j][d] = 1 iff drone d go from i to j
        beta = model.addMVar((nodes, nodes, self.drone), vtype=GRB.BINARY)
        model.addConstrs(beta[i, i, d] == 0 for i in range(nodes) for d in range(self.drone))

        # depot in and out
        model.addConstrs(np.ones((nodes,)) @ beta[0, :, d] == 1 for d in range(self.drone))
        model.addConstrs(np.ones((nodes,)) @ beta[:, -1, d] == 1 for d in range(self.drone))
        model.addConstrs(np.ones((nodes,)) @ beta[:, 0, d] == 0 for d in range(self.drone))
        model.addConstrs(np.ones((nodes,)) @ beta[-1, :, d] == 0 for d in range(self.drone))

        # in equals to out except for start and end depot
        model.addConstrs((np.ones((nodes,)) @ beta[:, j, d] == np.ones((nodes,)) @ beta[j, :, d]
                          for j in range(1, nodes - 1) for d in range(self.drone)), name='degree equation')

        # cannot fly from one city directly to another city
        model.addConstrs(beta[i][j][d] == 0 for i in range(1, len(group)) for j in range(1, len(group))
                         for d in range(self.drone))
        # can only take off to one city
        model.addConstrs(mask @ beta[i, :, d] <= 1 for i in range(nodes) for d in range(self.drone))
        model.addConstrs(mask @ beta[:, j, d] <= 1 for j in range(nodes) for d in range(self.drone))

        # in degree equals to out degree
        model.addConstrs(np.ones((nodes,)) @ beta[:, j, d] == np.ones((nodes,)) @ beta[j, :, d]
                         for j in range(1, nodes - 1) for d in range(self.drone))

        # visit all cities and ignore other cities
        model.addConstrs(np.ones((nodes,)) @ beta[:, c, :] @ np.ones((self.drone,)) == 1
                         for c in range(1, len(group)))
        model.addConstrs(np.ones((nodes,)) @ beta[c, :, :] @ np.ones((self.drone,)) == 1
                         for c in range(1, len(group)))

        model.addConstrs((beta[i, j, d] <= alpha[i, j] for i in range(len(group), nodes - 1)
                          for j in range(len(group), nodes) for d in range(self.drone)),
                         name='carried by truck')

        # continuous, time for arriving node i
        delta = model.addMVar((nodes,), vtype=GRB.CONTINUOUS)
        model.addConstr(delta[0] == 0)
        # delta[j] >= delta[i] + truck[i][j] - M * (1 - alpha[i][j])
        model.addConstrs(delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                         self.m * (1 - alpha[i, j]) for i in range(nodes - 1) for j in range(1, nodes))
        # delta[c] >= delta[i] + drone[i][c] - M * (1 - beta[i][c][d])
        model.addConstrs(delta[c] >= delta[i] + self.distance['drone'][_nodes[i]][_nodes[c]] / self.speed -
                         self.m * (1 - beta[i, c, d]) for i in range(nodes - 1) for c in range(1, len(group))
                         for d in range(self.drone))
        # delta[j] >= delta[c] + drone[j][c] - M * (1 - beta[c][j][d])
        model.addConstrs(delta[j] >= delta[c] + self.distance['drone'][_nodes[c]][_nodes[j]] / self.speed -
                         self.m * (1 - beta[c, j, d]) for j in range(1, nodes) for c in range(1, len(group))
                         for d in range(self.drone))
        # delta[j] >= delta[i] + truck[i][j] - M * (1 - beta[i][j][d])
        model.addConstrs(delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                         self.m * (1 - beta[i, j, d]) for i in range(len(group), nodes - 1)
                         for j in range(len(group), nodes) for d in range(self.drone))
        # delta[j] >= delta[i] + time_truck[i][j] - M * (2 - alpha[i][j] - beta[i][j][d])
        model.addConstrs((delta[j] >= delta[i] + self.distance['truck'][_nodes[i]][_nodes[j]] -
                          self.m * (2 - alpha[i, j] - beta[i, j, d]) for d in range(self.drone)
                          for i in range(nodes - 1) for j in range(1, nodes)), name='stupid constraint')
        # delta[j] >= delta[i] - M * (2 - beta[i][c][d] - beta[c][j][d])
        model.addConstrs(
            delta[j] >= delta[i] - self.m * (2 - beta[i, c, d] - beta[c, j, d]) for i in range(nodes - 1)
            for j in range(1, nodes) for d in range(self.drone) for c in range(1, len(group)))

        model.addConstrs(beta[i, j, d] <= alpha[i, j] for i in range(len(group), nodes - 1)
                         for j in range(len(group), nodes) for d in range(self.drone))

        # beta[i][c][d] <= sum_j([alpha[i][j]])
        model.addConstrs((beta[0][c][d] <= np.ones((nodes,)) @ alpha[0, :] for c in range(1, len(group))
                          for d in range(self.drone)), name='take off from depot')
        model.addConstrs(beta[i, c, d] <= np.ones((nodes,)) @ alpha[i, :] for i in range(len(group), nodes - 1)
                         for c in range(1, len(group)) for d in range(self.drone))
        model.addConstrs(beta[c, j, d] <= np.ones((nodes,)) @ alpha[:, j] for j in range(len(group), nodes)
                         for c in range(1, len(group)) for d in range(self.drone))

        phi = model.addMVar((nodes, self.drone), vtype=GRB.INTEGER)
        model.addConstrs(phi[i, d] - phi[j, d] <= nodes * (1 - beta[i, j, d]) - 1 for i in range(nodes - 1)
                         for j in range(1, nodes) for d in range(self.drone))
        model.addConstrs(phi[i, d] - phi[j, d] >= -nodes * (1 - beta[i, j, d]) - 1 for i in range(nodes - 1)
                         for j in range(1, nodes) for d in range(self.drone))
        model.addConstrs(phi[i, d] <= nodes for i in range(nodes) for d in range(self.drone))

        mu = model.addMVar((nodes,), vtype=GRB.INTEGER)
        model.addConstrs(mu[i] - mu[j] <= nodes * (1 - alpha[i][j]) - 1 for i in range(nodes - 1)
                         for j in range(1, nodes))
        model.addConstrs(mu[i] - mu[j] >= -nodes * (1 - alpha[i][j]) - 1 for i in range(nodes - 1)
                         for j in range(1, nodes))
        model.addConstrs(mu[i] <= nodes for i in range(nodes))

        model.setObjective(delta[nodes - 1], GRB.MINIMIZE)
        model.optimize()

        _truck_route = [i for i in range(len(_nodes) - 1) if np.ones((nodes,)) @ alpha[:, i].X > 0.5]
        _truck_route.sort(key=lambda x: mu[x].X)
        _truck_route = [depot] + [_nodes[i] for i in _truck_route] + [depot]
        truck_route = _truck_route

        drone_route = [[] for _ in range(self.drone)]
        for d in range(self.drone):
            for c in range(1, len(group)):
                for i in range(nodes):
                    if beta[i, c, d].X > 0.5:
                        for j in range(nodes):
                            if beta[c, j, d].X > 0.5:
                                drone_route[d].append([_nodes[i], _nodes[c], _nodes[j]])
        _drone_route = [[0] for _ in range(self.drone)]
        for d in range(self.drone):
            for i in range(nodes):
                for j in range(nodes):
                    if beta[i, j, d].X > 0.5:
                        _drone_route[d].append(j)
            _drone_route[d] = sorted(_drone_route[d], key=lambda x: phi[x, d].X)
            _drone_route[d] = [_nodes[node] for node in _drone_route[d]]
            # print(_drone_route)
        route = {'truck': truck_route, 'drone': drone_route, 'group': group}
        cost = {_nodes[i]: 0 for i in range(nodes)}
        for start, end in zip(route['truck'][:-1], route['truck'][1:]):
            for d in range(self.drone):
                for takeoff, city, land in route['drone'][d]:
                    if end == land:
                        cost[end] = max(cost[start] + self.distance['truck'][start][end],
                                        cost[takeoff] + (self.distance['drone'][takeoff][city] +
                                        self.distance['drone'][city][land]) / self.speed)
            cost[end] = max(cost[end], cost[start] + self.distance['truck'][start][end])
        return route, cost[_nodes[-1]]

    def get_solution(self):
        _, _, ksi = self.master_problem()
        epsilon = np.zeros((len(self.cities), len(self.depots)))
        for r in range(len(self.routes)):
            for _i in range(len(self.cities)):
                i = self.city_indices[self.cities[_i]]
                for j in range(len(self.nodes)):
                    for d in range(self.drone):
                        epsilon[_i][d] += ksi[r] * self.y[j][i][d][r]
        groups = [[depot] for depot in self.depots]
        index = np.argmax(epsilon, axis=1)
        for i in range(len(self.cities)):
            groups[index[i]].append(self.cities[i])
        for group in groups:
            route, cost = self.solve_optimal_pp(group)
            self.solution.append(route)
            self.cost = max(self.cost, cost)

    def solve(self):
        self.initial_solution()
        count, loop = 0, True
        while loop and count < 5:
            count += 1
            routes, theta, _ = self.master_problem()
            for route in routes:
                loop = loop and (self.pricing_problem(route, theta) > 0)
        self.get_solution()
        return self.solution, self.cost
