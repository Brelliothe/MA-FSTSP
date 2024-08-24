import math
import networkx as nx
import numpy as np


def euclidean(a, b):
    """return the Euclidean distance between two positions"""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def haversine(pos1, pos2):
    """return the distance of 2 position on earth in kilometers"""
    lon1, lat1 = pos1
    lon2, lat2 = pos2

    # Radius of the Earth in kilometers (mean radius)
    radius = 6371.0

    # Convert latitude and longitude from degrees to radians
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)

    x = (lon2_rad - lon1_rad) * math.cos(0.5 * (lat2_rad + lat1_rad))
    y = lat2_rad - lat1_rad

    # Distance in kilometers
    return radius * math.sqrt(x ** 2 + y ** 2)


def sign(number):
    return -1 if number < 0 else 1


def nearest_node(graph, location):
    distance = float('inf')
    nearest = None
    for node in graph.nodes:
        if haversine(location, graph.nodes[node]['pos']) < distance:
            distance = haversine(location, graph.nodes[node]['pos'])
            nearest = node
    assert nearest is not None, "no node closer than infinity, check the code"
    return nearest


def nearest_node_except_self(graph, name):
    distance, nearest = float('inf'), None
    for neighbor in graph.neighbors(name):
        # if haversine(graph.nodes[name]['pos'], graph.nodes[neighbor]['pos']) < distance:
        if graph.edges[name, neighbor]['weight'] < distance:
            nearest = neighbor
            distance = graph.edges[name, neighbor]['weight']
            # distance = haversine(graph.nodes[name]['pos'], graph.nodes[neighbor]['pos'])
    assert nearest is not None, "have no neighbor or all neighbors are further than infinity"
    assert nearest != name, "find self to be neighbor"
    return nearest


def base_convert(number, i, j):
    # convert (number)_i to (number)_j
    ans = []
    while number > 0:
        ans.append(number % i)
        number = number // i
    ans.reverse()
    return ans


def mst_partition(graph, depots, cities):
    """
    allocate the cities to salesman by partition the minimum spanning tree
    :param problem: (stations, depots, cities)
    :param lb: minimum cities a salesman should visit
    :param ub: maximum cities a salesman could visit
    :return: (stations, depots, cities) with matched depots and cities
    """
    # find the minimum spanning tree
    tree = nx.minimum_spanning_tree(graph)

    # turn the spanning tree into a rooted tree
    for i in tree.nodes:
        tree.nodes[i]['parent'] = -1

    def rooted_tree(node):
        # con represent the minimum cost in the subtree when the connected depot is its child
        # ncon represent the minimum cost in the subtree when the connected depot is not its child
        # for the depot node, treat itself as its child, set its ncon as -1 to be invalid
        cons, ncons, index, diff = [], [], [], []
        for n in tree.neighbors(node):
            if n != tree.nodes[node]['parent']:
                # indicator whether n will connect to a depot via its parent
                tree.nodes[n]['pcon'] = True
                tree.nodes[n]['parent'] = node
                con, ncon = rooted_tree(n)
                tree.nodes[n]['con'] = con
                tree.nodes[n]['ncon'] = ncon
                # n is a source
                if n in depots:
                    cons.append(con + tree[node][n]['weight'])
                    ncons.append(con)
                    diff.append(cons[-1] - ncons[-1])
                    index.append(n)
                # n is a target and has source inside the subtree
                elif con != -1:
                    cons.append(con + tree[node][n]['weight'])
                    ncons.append(min(ncon + tree[node][n]['weight'], con))
                    # the node will not connect to its parent if not connect source to its parent
                    diff.append(cons[-1] - ncons[-1])
                    index.append(n)
                    if ncons[-1] == con:
                        tree.nodes[n]['pcon'] = False
                # n do not have source within the subtree
                else:
                    ncons.append(ncon + tree[node][n]['weight'])
        if node in depots:
            # it is automatically connected to itself
            con = sum(ncons)
            ncon = -1
        elif len(diff) == 0:
            con = -1
            ncon = sum(ncons)
        else:
            id = np.argmin(np.array(diff))
            con = min(diff) + sum(ncons)
            ncon = sum(ncons)
            tree.nodes[node]['child'] = index[id]
        return con, ncon

    con, ncon = rooted_tree(depots[0])
    tree.nodes[depots[0]]['con'] = con
    tree.nodes[depots[0]]['ncon'] = ncon
    tree.nodes[depots[0]]['pcon'] = False

    def assign_group(node, value):
        if node in depots:
            # case 1: node is a depot, all children do not connect to both it and a depot inside the subtree
            # the node belongs to the group named by itself
            tree.nodes[node]['group'] = np.where(depots == node)[0].item()
            # for all neighbor nodes besides the parent
            for n in tree.neighbors(node):
                if n != tree.nodes[node]['parent']:
                    # if node n is also a depot
                    if n in depots:
                        # it belongs to the group named by itself
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    # if node n is a city but connects to 'node'
                    elif tree.nodes[n]['pcon']:
                        # assign its group to be node
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    # node n is a city and connects to some depot inside the subtree
                    else:
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

        # case 2: node is a city, and it connects to a depot inside the subtree
        elif value == tree.nodes[node]['con']:
            # find its child whose subtree contains the depot
            n = tree.nodes[node]['child']
            # if child n is a depot
            if n in depots:
                index = np.where(depots == n)[0].item()
                tree.nodes[node]['group'] = index
                tree.nodes[n]['group'] = index
                assign_group(n, tree.nodes[n]['con'])
            # if child n is a city
            else:
                tree.nodes[node]['group'] = assign_group(n, tree.nodes[n]['con'])
            # other neighbors besides child n and parent
            for n in tree.neighbors(node):
                if n != tree.nodes[node]['parent'] and n != tree.nodes[node]['child']:
                    # depot should not connect to it
                    if n in depots:
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    # city connect to 'node'
                    elif tree.nodes[n]['pcon']:
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    # city not connect to 'node'
                    else:
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])

        else:
            # node is a city and the connected depot is outside the subtree
            # a city is visited before so node has already been assigned a group
            for n in tree.neighbors(node):
                if n != tree.nodes[node]['parent']:
                    if n in depots:
                        tree.nodes[n]['group'] = np.where(depots == n)[0].item()
                        assign_group(n, tree.nodes[n]['con'])
                    elif tree.nodes[n]['pcon']:
                        tree.nodes[n]['group'] = tree.nodes[node]['group']
                        assign_group(n, tree.nodes[n]['ncon'])
                    else:
                        tree.nodes[n]['group'] = assign_group(n, tree.nodes[n]['con'])
        return tree.nodes[node]['group']

    # start from the first node
    assign_group(depots[0], tree.nodes[depots[0]]['con'])

    groups = {depot: [] for depot in depots}
    for node in tree.nodes:
        if node not in depots:
            groups[depots[tree.nodes[node]['group']]].append(node)
    return groups


def asymmetric_traveling_salesman_problem(graph, nodes_to_visit):
    new_graph = nx.Graph()
    for node in nodes_to_visit:
        new_graph.add_node(node)
        new_graph.add_node(node + 1000000)
    for node in nodes_to_visit:
        for _node in nodes_to_visit:
            new_graph.add_edge(node, _node + 1000000, weight=nx.dijkstra_path_length(graph, node, _node))
            new_graph.add_edge(node + 1000000, _node, weight=new_graph.edges[node, _node + 1000000]['weight'])
    for node in new_graph.nodes:
        for _node in new_graph.nodes:
            if new_graph.has_edge(node, _node):
                continue
            new_graph.add_edge(node, _node, weight=100000000)
    path = nx.approximation.christofides(new_graph)
    path = [node for node in path if node < 1000000]
    return path
