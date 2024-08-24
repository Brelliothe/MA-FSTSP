import networkx as nx
import sys
from src.lrmp import LinearRelaxedMasterProblem
from src.hc_vns import HillClimbingVariableNeighborhoodSearch
from src.fstsp import MultiAgentFlyingSidekickTSP
from src.lp import LinearProgramming
from problem import small_instance, multiagent_instance_on_manhattan, multiagent_instance_on_cambridge, manhattan
import numpy as np
import time
from tqdm import tqdm
from utils import haversine


def test_small_instance(num, size):
    graph, _depots, _cities, distance = small_instance(num, 20, 2, size)
    costs = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}
    times = {'lrmp': [], 'hc': [], 'stsp': [], 'lp': []}

    for i in tqdm(range(num)):
        depots, cities = _depots[i], _cities[i]
        model = LinearRelaxedMasterProblem(graph, depots, cities, distance, 2)
        start = time.time()
        solution, cost = model.solve()
        costs['lrmp'].append(cost)
        times['lrmp'].append(time.time() - start)
    print(f'LRMP gives solution with cost {sum(costs["lrmp"]) / num} in {sum(times["lrmp"]) / num}s')

    for i in tqdm(range(num)):
        depots, cities = _depots[i], _cities[i]
        model = HillClimbingVariableNeighborhoodSearch(graph, depots, cities, distance, 2, rounds=1000)
        start = time.time()
        solution, cost = model.solve()
        costs['hc'].append(cost)
        times['hc'].append(time.time() - start)
    print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}s')

    for i in tqdm(range(num)):
        depots, cities = _depots[i], _cities[i]
        model = MultiAgentFlyingSidekickTSP(graph, depots, cities, distance, 2)
        start = time.time()
        solution, cost = model.solve()
        costs['stsp'].append(cost)
        times['stsp'].append(time.time() - start)
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')

    for i in tqdm(range(num)):
        depots, cities = _depots[i], _cities[i]
        model = LinearProgramming(graph, depots, cities, distance, 2)
        start = time.time()
        solution, cost = model.solve()
        costs['lp'].append(cost)
        times['lp'].append(time.time() - start)
    print(f'LP gives solution with cost {sum(costs["lp"]) / num} in {sum(times["lp"]) / num}s')


def test_manhattan(num, size):
    graph, depots, cities, distance = multiagent_instance_on_manhattan(num, 5, size)
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    for i in tqdm(range(num)):
        depot, city = depots[i], cities[i]
        model = HillClimbingVariableNeighborhoodSearch(graph, depot, city, distance, 3, rounds=5000)
        start = time.time()
        solution, cost = model.solve()
        times['hc'].append(time.time() - start)
        costs['hc'].append(cost)
    print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"])/ num}')

    for i in tqdm(range(num)):
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(graph, depot, city, distance, 3, theta=(0.5, 0.5))
        start = time.time()
        solution, cost = model.solve()
        times['stsp'].append(time.time() - start)
        costs['stsp'].append(cost)
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')


def test_cambridge(num, size):
    graph, depots, cities, distance = multiagent_instance_on_cambridge(num, 10, size)
    costs = {'hc': [], 'stsp': [], 'lp': []}
    times = {'hc': [], 'stsp': [], 'lp': []}

    for i in tqdm(range(num)):
        depot, city = depots[i], cities[i]
        model = HillClimbingVariableNeighborhoodSearch(graph, depot, city, distance, 3, rounds=5000)
        start = time.time()
        solution, cost = model.solve()
        times['hc'].append(time.time() - start)
        costs['hc'].append(cost)
    print(f'Hill Climbing gives solution with cost {sum(costs["hc"]) / num} in {sum(times["hc"]) / num}')

    for i in tqdm(range(num)):
        depot, city = depots[i], cities[i]
        model = MultiAgentFlyingSidekickTSP(graph, depot, city, distance, 3, theta=(0.5, 0.5))
        start = time.time()
        solution, cost = model.solve()
        times['stsp'].append(time.time() - start)
        costs['stsp'].append(cost)
    print(f'Our algorithm gives solution with cost {sum(costs["stsp"]) / num} in {sum(times["stsp"]) / num}s')


def ablation_r():
    print('Studying the effect of radius limit')
    graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
    costs, times = [], []
    for r in range(5, 16, 2):
        costs.append([])
        times.append([])
        for i in tqdm(range(100)):
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, limit=r/10, theta=(0.5, 0.5))
            start = time.time()
            _, cost = model.solve()
            times[-1].append(time.time() - start)
            costs[-1].append(cost)
    np.save("r-time.npy", np.array(times))
    np.save("r-cost.npy", np.array(costs))


def ablation_speed():
    print('Studying the effect of speed')
    graph, depots, cities, distance = multiagent_instance_on_manhattan(20, 5, 100)
    costs, times = [], []
    for speed in [i / 30 for i in range(10, 120, 20)]:
        costs.append([])
        times.append(0)
        for i in tqdm(range(100)):
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 3, speed=speed)
            start = time.time()
            _, cost = model.solve()
            times[-1] += time.time() - start
            costs[-1].append(cost)
    np.save("speed-time.npy", np.array(times))
    np.save("speed-cost.npy", np.array(costs))


def ablation_k():
    print('studying the effect of drone number')
    for size in range(50, 160, 20):
        graph, depots, cities, distance = multiagent_instance_on_manhattan(10, 5, size)
        costs, times = [], []
        for i in tqdm(range(100)):
            model = MultiAgentFlyingSidekickTSP(graph, depots[i], cities[i], distance, 0)
            cost = model.solve_multiple_drones()
            costs.append(cost.copy())
        print(f'size {size} gives {np.mean(costs, axis=0)}')


def scale_cities():
    print('studying the scalability of fix depot case')
    graph = manhattan()
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
                'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                          for i in graph.nodes}}
    costs, times = [], []
    for num in range(120, 350, 40):
        costs.append([])
        times.append([])
        for _ in tqdm(range(100)):
            locations = np.random.choice(graph.nodes, num + 10, replace=False)
            model = MultiAgentFlyingSidekickTSP(graph, locations[:10], locations[10:], distance, 3)
            start = time.time()
            _, cost = model.solve()
            times[-1].append(time.time() - start)
            costs[-1].append(cost)
    np.save("city-time.npy", np.array(times))
    np.save("city-cost.npy", np.array(costs))


def scale_rates():
    print('studying the scalability of fix rates case')
    graph = manhattan()
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
                'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                          for i in graph.nodes}}
    costs, times = [], []
    for num in range(3, 21, 3):
        costs.append([])
        times.append([])
        for _ in tqdm(range(100)):
            locations = np.random.choice(graph.nodes, num * (1 + 20), replace=False)
            model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 'b3', 3)
            start = time.time()
            _, cost = model.solve()
            times[-1].append(time.time() - start)
            costs[-1].append(cost)
    np.save("rates-time.npy", np.array(times))
    np.save("rates-cost.npy", np.array(costs))


def scale_depots():
    print('studying the scalability of fix cities case')
    graph = manhattan()
    distance = {'truck': dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight')),
                'drone': {i: {j: haversine(graph.nodes[i]['pos'], graph.nodes[j]['pos']) for j in graph.nodes}
                          for i in graph.nodes}}
    costs, times = [], []
    for num in range(5, 16, 2):
        costs.append([])
        times.append([])
        for _ in tqdm(range(100)):
            locations = np.random.choice(graph.nodes, num + 150, replace=False)
            model = MultiAgentFlyingSidekickTSP(graph, locations[:num], locations[num:], distance, 3)
            start = time.time()
            _, cost = model.solve()
            times[-1].append(time.time() - start)
            costs[-1].append(cost)
    np.save("depots-time.npy", np.array(times))
    np.save("depots-cost.npy", np.array(costs))


if __name__ == '__main__':
    for size in [5, 10, 15]:
        test_small_instance(100, size)
    for size in [50, 100, 150]:
        test_manhattan(100, size)
        test_cambridge(100, size)
    ablation_r()
    ablation_speed()
    ablation_k()
    scale_cities()
    scale_rates()
    scale_depots()
