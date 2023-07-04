import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np

from numgrids import Interpolator


class Plotter:

    def __init__(self, grid):
        self.grid = grid
        self.fig = None
        self.ax = None

    def plot(self, f, *coords):

        num_points = None
        for c in coords:
            if hasattr(c, "__len__"):
                num_points = len(c)
                break

        parsed_coords = []
        for c in coords:
            if hasattr(c, "__len__"):
                parsed_coords.append(c)
            else:
                parsed_coords.append([c] * num_points)
        coords = parsed_coords

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.subplots(1, 1)

        inter = Interpolator(self.grid, f)
        coords = np.array(coords).T
        values = inter(coords)

        self.ax.plot(values)

    def show(self):
        self.fig.show()

