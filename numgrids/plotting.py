import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot(*objects):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('equal')
    for obj in objects:
        data = obj.plot_data()
        plt.fill(data[:, 0], data[:, 1])
    return ax
