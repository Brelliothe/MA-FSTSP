import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
from baseline import Baseline


class LinearProgramming(Baseline):
    def __init__(self, graph, depots, cities, distance, drone, limit=1.5, speed=1.6):
        super().__init__(graph, depots, cities, distance, drone, limit, speed)
        self.solution = []
        self.cost = 0
        self.subgraph = nx.Graph()
        self.truck_graph = nx.Graph()
        self.drone_graph = nx.Graph()
        self.boundary = {}

    def induce(self):
        for city in self.cities:
            self.subgraph.add_node(city)
            self.boundary[city] = []
            for node in self.graph.nodes:
                if self.distance['drone'][node][city] <= self.limit / 2:
                    for _node in nx.neighbors(self.graph, node):
                        if self.distance['drone'][_node][city] > self.limit / 2:
                            self.subgraph.add_node(node)
                            self.boundary[city].append(node)
                            break
        for depot in self.depots:
            self.subgraph.add_node(depot)
            self.boundary[depot] = [depot]
        for u in self.subgraph.nodes:
            for v in self.subgraph.nodes:
                self.subgraph.add_edge(u, v, weight=self.distance['truck'][u][v])

    def solve(self):
        # solve the set matsp with radius r/2
        self.induce()
        num_nodes = len(self.subgraph.nodes)
        nodes = list(self.subgraph.nodes)
        depots = [i for i in range(num_nodes) if nodes[i] in self.depots]
        cities = [i for i in range(num_nodes) if nodes[i] in self.cities]
        locs = depots + cities
        num_locs = len(locs)
        sets = {i: [nodes.index(node) for node in self.boundary[nodes[locs[i]]]] for i in range(num_locs)}
        weights = np.array([[self.subgraph[nodes[u]][nodes[v]]['weight'] for v in range(num_nodes)]
                            for u in range(num_nodes)])
        model = gp.Model('LP')
        model.setParam('OutputFlag', 0)
        # first do a city + depot matsp
        city_route = model.addMVar((num_locs, num_locs), vtype=GRB.BINARY)
        # in-deg = out-deg
        model.addConstrs(np.ones((num_locs,)) @ city_route[:, v] == np.ones((num_locs,)) @ city_route[v, :]
                         for v in range(num_locs))
        # each city is visited exactly once
        model.addConstrs(city_route[:, c] @ np.ones((num_locs,)) == 1 for c in range(len(depots), num_locs))
        model.addConstrs(city_route[c, :] @ np.ones((num_locs,)) == 1 for c in range(len(depots), num_locs))
        # no self loop except for depots
        model.addConstrs(city_route[v, v] == 0 for v in range(len(depots), num_locs))
        # flow constraints as GG formulation
        flow = model.addMVar((num_locs, num_locs), vtype=GRB.CONTINUOUS)
        model.addConstr(flow[:, :] <= num_locs * city_route[:, :])
        model.addConstr(gp.quicksum([flow[d, :] @ np.ones((num_locs,)) for d in range(len(depots))]) == len(cities))
        model.addConstrs(flow[:, d] @ np.ones((num_locs,)) == 0 for d in range(len(depots)))
        model.addConstrs(flow[:, v] @ np.ones((num_locs,)) - flow[v, :] @ np.ones((num_locs,)) >= 1
                         for v in range(len(depots), num_locs))
        # route represent the actual route
        route = model.addMVar((num_nodes, num_nodes), vtype=GRB.BINARY)
        model.addConstrs(gp.quicksum([route[s, t] for s in sets[u] for t in sets[v]]) >= city_route[u, v]
                         for u in range(num_locs) for v in range(num_locs))
        # routes are cycles
        model.addConstrs(route[v, v] == 0 for v in range(num_nodes) if v not in depots)
        model.addConstrs(route[:, v] @ np.ones((num_nodes,)) == route[v, :] @ np.ones((num_nodes,))
                         for v in range(num_nodes))
        # all nodes should connect to depots
        goods = model.addMVar((num_nodes, num_nodes), vtype=GRB.CONTINUOUS)
        model.addConstr(goods[:, :] <= num_nodes * route[:, :])
        model.addConstrs(goods[:, u] @ np.ones((num_nodes,)) - goods[u, :] @ np.ones((num_nodes,)) ==
                         route[:, u] @ np.ones((num_nodes,)) for u in range(num_nodes) if u not in depots)
        obj = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(obj >= (weights * route).sum())
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()
        if model.Status == GRB.TIME_LIMIT:
            self.solution = []
            self.cost = model.getAttr(gp.GRB.Attr.ObjBound)
        else:
            # no need to return the solution, just update the cost
            self.cost = model.objVal
        return self.solution, self.cost
