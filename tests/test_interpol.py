import unittest

import numpy as np

from numgrids import Axis, AxisType, Grid
from numgrids.interpol import Interpolant


class TestInterpolation(unittest.TestCase):

    def test_interpol1d(self):
        grid = Grid(Axis.of_type(AxisType.EQUIDISTANT, 50, 0, 1))
        x = grid.coords
        f = x**2
        expected = 0.5**2
        inter = Interpolant(grid, f)
        actual = inter(0.5)
        self.assertAlmostEqual(actual, expected)

    def test_interpol2d(self):
        grid = Grid(
            Axis.of_type(AxisType.EQUIDISTANT, 50, 0, 1),
            Axis.of_type(AxisType.CHEBYSHEV, 50, 0, 1)
        )
        X, Y = grid.meshed_coords
        f = X ** 2 + Y**2
        expected = 0.5 ** 2 + 0.5**2
        inter = Interpolant(grid, f)
        actual = inter((0.5, 0.5))
        self.assertAlmostEqual(actual, expected)

    def test_interpol_polar(self):
        grid = Grid(
            Axis.of_type(AxisType.CHEBYSHEV, 50, 0, 1),
            Axis.of_type(AxisType.EQUIDISTANT_PERIODIC, 50, 0, 2*np.pi)
        )
        R, Phi = grid.meshed_coords
        f = R**2 * np.sin(Phi)
        expected = 0.5**2 * np.sin(np.pi/4)
        interp = Interpolant(grid, f)
        actual = interp((0.5, np.pi / 4))
        self.assertAlmostEqual(actual, expected, places=6)

