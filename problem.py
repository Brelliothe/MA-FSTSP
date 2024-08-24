import json
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pickle
from utils import haversine


def manhattan():
    """read the manhattan road map from file and return it as a networkx graph"""
    g = nx.MultiDiGraph(nx.read_graphml('nyc.graphml'))
    manhattan_graph = nx.MultiDiGraph()
    index = {node: i for node, i in zip(g.nodes, range(len(g.nodes)))}
    for node in g.nodes:
        manhattan_graph.add_node(index[node], pos=[float(g.nodes[node]['lon']), float(g.nodes[node]['lat'])])
    for edge in g.edges:
        start = manhattan_graph.nodes[index[edge[0]]]['pos']
        end = manhattan_graph.nodes[index[edge[1]]]['pos']
        manhattan_graph.add_edge(index[edge[0]], index[edge[1]], weight=haversine(start, end))
    if not os.path.isfile('datasets/manhattan.json'):
        print('=============preparing pairwise data=================')
        lengths = dict(nx.all_pairs_dijkstra_path_length(manhattan_graph, weight='weight'))
        pairwise_distances = [[lengths[i][j] for j in manhattan_graph.nodes] for i in manhattan_graph.nodes]
        with open('datasets/manhattan.json', 'w') as f:
            json.dump(pairwise_distances, f)
    return manhattan_graph


def cambridge():
    """read the cambridge road map from osm and return it as a networkx graph"""
    place_name = "Boston, MA, USA"
    graph = ox.graph_from_place(place_name, network_type='drive')
    nodes = max(nx.strongly_connected_components(graph), key=len)
    index = {node: i for node, i in zip(list(nodes), range(len(nodes)))}
    cambridge_graph = nx.MultiDiGraph()
    for node in list(nodes):
        cambridge_graph.add_node(index[node], pos=[float(graph.nodes[node]['x']), float(graph.nodes[node]['y'])])
    for edge in graph.edges:
        if edge[0] in nodes and edge[1] in nodes:
            start = cambridge_graph.nodes[index[edge[0]]]['pos']
            end = cambridge_graph.nodes[index[edge[1]]]['pos']
            cambridge_graph.add_edge(index[edge[0]], index[edge[1]], weight=haversine(start, end))
    return cambridge_graph


def random_multiagent_instance(graph, num_depots, num_destinations):
    np.random.seed(0)
    assert len(graph.nodes) > num_depots + num_destinations, \
        f"impossible to sample {num_depots + num_destinations} locations from {len(graph.nodes)} nodes"
    assert num_depots > 1, f"fewer than 2 depots, try to use random_instance function to generate for single agent"
    locations = np.random.choice(graph.nodes, size=num_depots+num_destinations)
    return graph, locations[:num_depots], locations[num_depots:]


def small_instance(num, nodes, depots, cities):
    np.random.seed(0)
    # select a subset of the whole manhattan graph
    graph = manhattan()
    # randomly select a node
    node = np.random.choice(graph.nodes, 1).item()
    # do the bfs to find a set of nodes:
    _nodes = [node]
    subgraph = nx.DiGraph()
    subgraph.add_node(node, pos=graph.nodes[node]['pos'])
    while subgraph.number_of_nodes() < nodes:
        neighbors = graph.neighbors(_nodes.pop(0))
        for n in neighbors:
            if not subgraph.has_node(n):
                _nodes.append(n)
                subgraph.add_node(n, pos=graph.nodes[n]['pos'])
            if subgraph.number_of_nodes() >= nodes:
                break
    assert subgraph.number_of_nodes() == nodes, 'wrong number of nodes, check the code'
    for start in subgraph.nodes:
        for end in subgraph.nodes:
            if graph.has_edge(start, end):
                subgraph.add_edge(start, end, weight=graph.edges[start, end, 0]['weight'])
                subgraph.add_edge(end, start, weight=graph.edges[start, end, 0]['weight'])
    # compute the distance
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(subgraph, weight='weight')),
                'drone': {i: {j: haversine(subgraph.nodes[i]['pos'], subgraph.nodes[j]['pos'])
                              for j in subgraph.nodes} for i in subgraph.nodes}}
    _depots, _cities = [], []
    for _ in range(num):
        locations = np.random.choice(subgraph.nodes, depots + cities, replace=False)
        _depots.append(locations[:depots])
        _cities.append(locations[depots:])
    return subgraph, _depots, _cities, distance


def multiagent_instance_on_manhattan(num, depots, cities):
    np.random.seed(0)
    graph = manhattan()
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
                'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                          for i in graph.nodes}}
    _depots, _cities = [], []
    for _ in range(num):
        locations = np.random.choice(graph.nodes, depots + cities, replace=False)
        np.random.shuffle(locations)
        _depots.append(locations[:depots])
        _cities.append(locations[depots:])
    return graph, _depots, _cities, distance


def multiagent_instance_on_cambridge(num, depots, cities):
    np.random.seed(0)
    graph = cambridge()
    if os.path.isfile('datasets/cambridge_all_pair_road_distance.pkl'):
        with open('datasets/cambridge_all_pair_road_distance.pkl', 'rb') as f:
            distance = pickle.load(f)
    else:
        distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
                    'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                              for i in graph.nodes}}
        with open('datasets/cambridge_all_pair_road_distance.pkl', 'wb') as f:
            pickle.dump(distance, f)
    _depots, _cities = [], []
    for _ in range(num):
        locations = np.random.choice(graph.nodes, depots + cities, replace=False)
        _depots.append(locations[:depots])
        _cities.append(locations[depots:])
    return graph, _depots, _cities, distance
