from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from numgrids.interpol import Interpolator

if TYPE_CHECKING:
    from numgrids.grids import Grid


class Plotter:
    """Simple plotting utility for visualizing meshed functions on a grid."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    def plot(self, f: NDArray, *coords) -> None:
        """Plot a meshed function along specified coordinates."""

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

    def show(self) -> None:
        """Display the current plot."""
        if self.fig is not None:
            self.fig.show()

