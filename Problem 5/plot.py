import operator
import matplotlib.pyplot as plt


def plot_graph(points, path: list):
    x = []
    y = []
    for a, b in points:
        x.append(a)
        y.append(b)
    y = list(map(operator.sub, [max(y) for i in range(len(points))], y))
    plt.plot(x, y, 'co')
    for i, j in zip(path + path[-1:], path[1:] + path[:1]):
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='r', length_includes_head=True)
    plt.show()
