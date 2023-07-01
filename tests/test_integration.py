import unittest

import numpy as np
import numpy.testing as npt

from numgrids import Grid, Axis, AxisType
from numgrids.integration import Integral


class TestIntegral(unittest.TestCase):

    def test_integral_1d_findiff(self):
        grid = Grid(
            Axis(AxisType.EQUIDISTANT, 100, -1, 1)
        )

        x = grid.coords
        f = np.cos(x)

        expected = np.sin(1) - np.sin(-1)

        I = Integral(grid)

        actual = I(f)

        npt.assert_array_almost_equal(actual, expected)

    def test_integral_2d_findiff(self):
        grid = Grid(
            Axis(AxisType.EQUIDISTANT, 100, -1, 1),
            Axis(AxisType.CHEBYSHEV, 100, 0, 1)
        )

        X, Y = grid.meshed_coords
        x, y = grid.coords
        f = np.cos(X) + Y

        I = Integral(grid)
        # \int f(x, y) dx dy
        expected = np.sin(1) - np.sin(-1) + 0.5 * (x[-1] - x[0])

        actual = I(f)
        npt.assert_array_almost_equal(actual, expected)

    def test_integral_3d_findiff(self):
        grid = Grid(
            Axis(AxisType.EQUIDISTANT, 50, -1, 1),
            Axis(AxisType.EQUIDISTANT, 50, -1, 1),
            Axis(AxisType.EQUIDISTANT, 50, -1, 1),
        )

        X, Y, Z = grid.meshed_coords
        f = np.sin(X) ** 2 + np.sin(Y) ** 2 + np.sin(Z) ** 2

        I = Integral(grid)

        expected = -12 * np.sin(1) * np.cos(1) + 12

        actual = I(f)
        npt.assert_array_almost_equal(actual, expected, decimal=5)

    def test_integral_3d_findiff_cheb(self):
        grid = Grid(
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
        )

        X, Y, Z = grid.meshed_coords
        x, y, z = grid.coords
        f = np.sin(X) ** 2 + np.sin(Y) ** 2 + np.sin(Z) ** 2

        I = Integral(grid)

        expected = -12 * np.sin(1) * np.cos(1) + 12

        actual = I(f)
        npt.assert_array_almost_equal(actual, expected)

    def test_fft_integration(self):

        grid = Grid(
            Axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 4*np.pi),
        )

        x = grid.coords
        f = np.cos(x)**2

        I = Integral(grid)

        expected = 2*np.pi
        actual = I(f)

        self.assertAlmostEqual(actual, expected)
