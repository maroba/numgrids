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
    """Plotting utility for visualizing meshed functions on a grid.

    Wraps matplotlib to provide quick line plots of interpolated
    function values along user-specified coordinate slices.
    """

    def __init__(self, grid: Grid) -> None:
        """Create a plotter for the given grid.

        Parameters
        ----------
        grid : Grid
            The grid on which the data lives.
        """
        self.grid = grid
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

    def plot(self, f: NDArray, *coords) -> None:
        """Plot interpolated function values along a coordinate path.

        Parameters
        ----------
        f : NDArray
            Meshed function values on the grid.
        *coords
            Coordinate arrays or scalar values defining the path to plot.
            At least one coordinate must be an array; scalar coordinates
            are broadcast to match.
        """

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

