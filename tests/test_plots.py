import unittest

from numgrids import AxisType, Grid, create_axis
from numgrids.plots import Plotter


class TestPlotter(unittest.TestCase):

    def test_plot(self):
        axis = create_axis(AxisType.EQUIDISTANT, 30, -1, 1)
        grid = Grid(axis, axis)

        X, Y = grid.meshed_coords
        f = X**2 + Y**2

        plotter = Plotter(grid)

        plotter.plot(f, axis.coords, 0)
        self.assertIsNotNone(plotter.ax)


