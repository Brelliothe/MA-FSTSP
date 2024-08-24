import folium
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.fstsp import MultiAgentFlyingSidekickTSP
from problem import multiagent_instance_on_manhattan


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
fontsize = 18
parameters = {
    'font.family': 'cmr10',
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize,
    'axes.axisbelow': True
}
plt.rcParams.update(parameters)
colors = sns.color_palette()


def plot_graph(graph):
    plt.scatter([graph.nodes[node]['pos'][0] for node in graph.nodes],
                [graph.nodes[node]['pos'][1] for node in graph.nodes], s=2)
    for edge in graph.edges:
        plt.plot([graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[1]]['pos'][0]],
                 [graph.nodes[edge[0]]['pos'][1], graph.nodes[edge[1]]['pos'][1]], color='black', linewidth=1)


def plot_truck_solution(graph, solution):
    for start, end in zip(solution[:-1], solution[1:]):
        path = nx.shortest_path(G=graph, source=start, target=end, weight='weight')
        for edge_start, edge_end in zip(path[:-1], path[1:]):
            plt.plot([graph.nodes[edge_start]['pos'][0], graph.nodes[edge_end]['pos'][0]],
                     [graph.nodes[edge_start]['pos'][1], graph.nodes[edge_end]['pos'][1]], color='red')


def plot_multiagent_solution(graph, solution, depots, cities):
    plot_graph(graph)
    plt.scatter([graph.nodes[node]['pos'][0] for node in depots],
                [graph.nodes[node]['pos'][1] for node in depots], s=20, c='red', marker='o')
    plt.scatter([graph.nodes[node]['pos'][0] for node in cities],
                [graph.nodes[node]['pos'][1] for node in cities], s=20, c='blue', marker='o')
    for node in np.concatenate((depots, cities)):
        plt.text(graph.nodes[node]['pos'][0], graph.nodes[node]['pos'][1], node)
    for route in solution:
        plot_truck_solution(graph, route['truck'])
        for drone_route in route['drone']:
            for route_d in drone_route:
                for start, end in zip(route_d[:-1], route_d[1:]):
                    plt.plot([graph.nodes[start]['pos'][0], graph.nodes[end]['pos'][0]],
                             [graph.nodes[start]['pos'][1], graph.nodes[end]['pos'][1]], linestyle='-', color='green')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_r():
    times = np.load("r-time.npy")
    costs = np.load("r-cost.npy")
    fig, ax1 = plt.subplots()
    box = ax1.boxplot(costs.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=False)
    for patch in box['boxes']:
        patch.set_facecolor(colors[0])
    ax1.set_xlabel('Distance Limit')
    ax1.set_ylabel('Cost (Boxes)')

    times = times
    ax2 = ax1.twinx()
    ax2.plot(range(1, 7), np.mean(times, axis=1), marker='d', markersize=10)
    ax2.set_ylabel('Time(s) (Line)')
    ax2.set_yscale('log')
    plt.xticks(ticks=range(1, 7), labels=[f'{i / 10:.1f}' for i in range(5, 16, 2)])
    plt.tight_layout()
    plt.savefig('r.pdf')
    plt.show()


def plot_speed():
    times = np.load("speed-time.npy")
    costs = np.load("speed-cost.npy")
    fig, ax1 = plt.subplots()
    ax1.boxplot(costs.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    ax1.set_xlabel('Ratio of Speed')
    ax1.set_ylabel('Cost (Boxes)')

    times = times
    ax2 = ax1.twinx()
    ax2.plot(range(1, 7), times / 25, marker='d', color=colors[0], markersize=10)
    ax2.set_ylabel('Time(s) (Line)')
    plt.xticks(ticks=range(1, 7), labels=[f'{i / 30:.2f}' for i in range(10, 120, 20)])
    plt.tight_layout()
    plt.savefig('speed.pdf')
    plt.show()


def plot_k():
    costs = np.load('k-cost.npy')
    for i in range(5):
        size = 50 + 20 * i
        cost = costs[i]
        print(f'at size {size}, the average cost is {np.mean(cost, axis=0)}')


def plot_cities():
    times = np.load('city-time.npy')[2:]
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[120 + 40 * i for i in range(6)])
    plt.text(0.05, 0.9, "$|\mathcal{P}|=10$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Customers')
    plt.tight_layout()
    plt.savefig('city.pdf')
    plt.show()


def plot_rates():
    times = np.load('rates-time.npy')
    # times[times > 300] = 300
    # times = times[4:]
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[20 * i for i in range(3, 21, 3)])
    plt.text(0.05, 0.9, "$|\mathcal{C}|/|\mathcal{P}|=20$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Customers')
    plt.tight_layout()
    plt.savefig('rates.pdf')
    plt.show()


def plot_depots():
    times = np.load('depots-time.npy')
    plt.grid()
    plt.boxplot(times.T, patch_artist=True, boxprops=dict(facecolor='C0'), showfliers=True)
    plt.xticks(ticks=range(1, 7), labels=[5 + 2 * i for i in range(6)])
    plt.text(0.75, 0.9, "$|\mathcal{C}|=150$", fontsize=22, transform=plt.gca().transAxes, verticalalignment='top')
    plt.ylabel('Time(s)')
    plt.xlabel('Depots')
    plt.tight_layout()
    plt.savefig('depots.pdf')
    plt.show()


def plot_accelerate():
    from problem import small_instance
    from matplotlib.patches import Circle
    from utils import euclidean
    graph, depots, cities, distance = small_instance(10, 50, 1, 2)
    cities = [773, 994]
    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, c=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, c=colors[1], marker='o')

    blue_region = Circle((graph.nodes[cities[0]]['pos'][0], graph.nodes[cities[0]]['pos'][1]),
                         0.007, color=colors[0], alpha=0.4)
    red_region = Circle((graph.nodes[cities[1]]['pos'][0], graph.nodes[cities[1]]['pos'][1]),
                        0.007, color=colors[1], alpha=0.4)
    ax = plt.gca()
    ax.add_patch(blue_region)
    ax.add_patch(red_region)
    for node in graph.nodes:
        if euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos']) < 0.007:
            if euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos']) < 0.007:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, c=colors[4], marker='o')
            else:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, c=colors[0], marker='o')
        else:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, c=colors[1], marker='o')
    # x_min = min([graph.nodes[node]['pos'][0] for node in graph.nodes])
    # x_max = max([graph.nodes[node]['pos'][0] for node in graph.nodes])
    # y_min = min([graph.nodes[node]['pos'][1] for node in graph.nodes])
    # y_max = max([graph.nodes[node]['pos'][1] for node in graph.nodes])
    x_min, x_max, y_min, y_max = -73.9850867, -73.970341, 40.751748, 40.7604633
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('overlap.pdf')
    plt.show()

    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, c=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, c=colors[1], marker='o')
    blue_vertices = []
    red_vertices = []
    for node in graph.nodes:
        d_1 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos'])
        d_2 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos'])
        if d_1 < d_2:
            blue_vertices.append(graph.nodes[node]['pos'])
        else:
            red_vertices.append(graph.nodes[node]['pos'])
        if d_1 < 0.007 and d_2 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, c=colors[0] if d_1 < d_2 else colors[1], marker='o')
        elif d_1 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, c=colors[0], marker='o')
        elif d_2 < 0.007:
            plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]],
                        s=50, c=colors[1], marker='o')
    import alphashape
    blue_vertices = alphashape.alphashape(blue_vertices, 0.1)
    red_vertices = alphashape.alphashape(red_vertices, 0.1)
    ax = plt.gca()
    x, y = blue_vertices.exterior.xy
    ax.fill(x, y, facecolor=colors[0], edgecolor=colors[0], alpha=0.4)
    x, y = red_vertices.exterior.xy
    ax.fill(x, y, facecolor=colors[1], edgecolor=colors[1], alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('overlap-2.pdf')
    plt.show()

    plot_graph(graph)
    plt.scatter([graph.nodes[cities[0]]['pos'][0]], [graph.nodes[cities[0]]['pos'][1]],
                s=200, c=colors[0], marker='o')
    plt.scatter([graph.nodes[cities[1]]['pos'][0]], [graph.nodes[cities[1]]['pos'][1]],
                s=200, c=colors[1], marker='o')
    blue_vertices, red_vertices = [], []
    for node in graph.nodes:
        d_1 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[0]]['pos'])
        d_2 = euclidean(graph.nodes[node]['pos'], graph.nodes[cities[1]]['pos'])
        if d_1 < d_2:
            blue_vertices.append(node)
        else:
            red_vertices.append(node)
    for node in blue_vertices:
        for neighbor in graph.neighbors(node):
            if neighbor in red_vertices:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, c=colors[0], marker='o')
                break
    for node in red_vertices:
        for neighbor in graph.neighbors(node):
            if neighbor in blue_vertices:
                plt.scatter([graph.nodes[node]['pos'][0]], [graph.nodes[node]['pos'][1]], s=50, c=colors[1], marker='o')
                break
    blue_vertices = [graph.nodes[node]['pos'] for node in blue_vertices]
    red_vertices = [graph.nodes[node]['pos'] for node in red_vertices]
    blue_boundary = alphashape.alphashape(blue_vertices, 0.1)
    red_boundary = alphashape.alphashape(red_vertices, 0.1)
    plt.scatter(np.array(blue_boundary.exterior.coords)[:, 0], np.array(blue_boundary.exterior.coords)[:, 1],
                s=50, c=colors[0])
    plt.scatter(np.array(red_boundary.exterior.coords)[:, 0], np.array(red_boundary.exterior.coords)[:, 1],
                s=50, c=colors[1])
    ax = plt.gca()
    x, y = blue_boundary.exterior.xy
    ax.fill(x, y, facecolor=colors[0], edgecolor=colors[0], alpha=0.4)
    x, y = red_boundary.exterior.xy
    ax.fill(x, y, facecolor=colors[1], edgecolor=colors[1], alpha=0.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('overlap-3.pdf')
    plt.show()


def plot_example():
    graph, depots, cities, distance = multiagent_instance_on_manhattan(1, 2, 20)
    model = MultiAgentFlyingSidekickTSP(graph, depots[0], cities[0], distance, 2, limit=0.8)

    # plot the map
    m = folium.Map(location=[40.77, -73.96], zoom_start=14, tiles='Cartodb Positron')
    for edge in graph.edges:
        lon0, lat0 = graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[0]]['pos'][1]
        lon1, lat1 = graph.nodes[edge[1]]['pos'][0], graph.nodes[edge[1]]['pos'][1]
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1).add_to(m)

    # draw the partition
    model.set_nn(0.5)
    groups = model.groups
    color = {depots[0][0]: 'blue', depots[0][1]: 'red'}
    for depot in groups.keys():
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(m)
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          color=color[depot], weight=2, radius=40, fill=False).add_to(m)
    m.save('map.html')

    n = folium.Map(location=[40.77, -73.96], zoom_start=14, tiles='Cartodb Positron')
    o = folium.Map(location=[40.77, -73.96], zoom_start=14, tiles='Cartodb Positron')
    for edge in graph.edges:
        lon0, lat0 = graph.nodes[edge[0]]['pos'][0], graph.nodes[edge[0]]['pos'][1]
        lon1, lat1 = graph.nodes[edge[1]]['pos'][0], graph.nodes[edge[1]]['pos'][1]
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1, opacity=0.5).add_to(n)
        folium.PolyLine(locations=[[lat0, lon0], [lat1, lon1]], color='black', weight=1, opacity=0.5).add_to(o)

    for depot in groups.keys():
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(n)
        folium.Circle(location=[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]], color=color[depot],
                      fill_color=color[depot], fill_opacity=1, radius=40).add_to(o)
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          color=color[depot], weight=2, radius=40, fill=False).add_to(n)

    # draw the set TSP
    for depot in model.depots:
        for city in groups[depot]:
            folium.Circle(location=[graph.nodes[city]['pos'][1], graph.nodes[city]['pos'][0]],
                          radius=400, color=color[depot], weight=0.5, fill_color=color[depot], fill_opacity=0.2).add_to(
                o)
        solution, route = model.single_solution(depot, 0.5)
        locations = [[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]]]
        for start, end in zip(route[:-1], route[1:]):
            path = nx.dijkstra_path(graph, start, end, weight='weight')
            for node in path[1:]:
                locations.append([graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]])
        folium.PolyLine(locations=locations, color=color[depot], weight=5).add_to(o)

        locations = [[graph.nodes[depot]['pos'][1], graph.nodes[depot]['pos'][0]]]
        for start, end in zip(solution['truck'][:-1], solution['truck'][1:]):
            path = nx.dijkstra_path(graph, start, end, weight='weight')
            for node in path[1:]:
                locations.append([graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]])
        folium.PolyLine(locations=locations, color=color[depot], weight=5).add_to(n)
        for drone_route in solution['drone']:
            folium.PolyLine(
                locations=[[graph.nodes[node]['pos'][1], graph.nodes[node]['pos'][0]] for node in drone_route],
                color='green', weight=5).add_to(n)
    o.save('tsp.html')
    n.save('solution.html')


if __name__ == '__main__':
    plot_speed()
    plot_k()
    plot_cities()
    plot_rates()
    plot_depots()
    plot_r()
    plot_accelerate()
    plot_example()
