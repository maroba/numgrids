import unittest

import numpy as np
import numpy.testing as npt

from numgrids import Axis, AxisType, Grid
from numgrids.interpol import Interpolator


class TestInterpolation(unittest.TestCase):

    def test_interpol1d(self):
        grid = Grid(Axis(AxisType.EQUIDISTANT, 50, 0, 1))
        x = grid.coords
        f = x ** 2
        expected = 0.5 ** 2
        inter = Interpolator(grid, f)
        actual = inter(0.5)
        self.assertAlmostEqual(actual, expected)

    def test_interpol2d(self):
        grid = Grid(
            Axis(AxisType.EQUIDISTANT, 50, 0, 1),
            Axis(AxisType.CHEBYSHEV, 50, 0, 1)
        )
        X, Y = grid.meshed_coords
        f = X ** 2 + Y ** 2
        expected = 0.5 ** 2 + 0.5 ** 2
        inter = Interpolator(grid, f)
        actual = inter((0.5, 0.5))
        self.assertAlmostEqual(actual, expected)

    def test_interpol_polar(self):
        grid = Grid(
            Axis(AxisType.CHEBYSHEV, 50, 0, 1),
            Axis(AxisType.EQUIDISTANT_PERIODIC, 50, 0, 2 * np.pi)
        )
        R, Phi = grid.meshed_coords
        f = R ** 2 * np.sin(Phi)
        expected = 0.5 ** 2 * np.sin(np.pi / 4)
        interp = Interpolator(grid, f)
        actual = interp((0.5, np.pi / 4))
        self.assertAlmostEqual(actual, expected, places=6)

    def test_interpol_many(self):
        grid = Grid(
            Axis(AxisType.EQUIDISTANT, 30, 0, 1),
            Axis(AxisType.CHEBYSHEV, 30, 0, 1),
            Axis(AxisType.CHEBYSHEV, 30, 0, 1)
        )
        X, Y, Z = grid.meshed_coords

        f_ = lambda x, y, z: x**2 + y**2 + z**2
        f = f_(X, Y, Z)

        inter = Interpolator(grid, f)

        t = np.linspace(0, 1, 3)
        points = zip(t, t ** 2, t)

        actual = inter(points)
        expected = [f_(0, 0, 0), f_(0.5, 0.25, 0.5), f_(1, 1, 1)]
        npt.assert_array_almost_equal(actual, expected)

    def test_interpol_grid(self):
        fine_grid = Grid(
            Axis(AxisType.EQUIDISTANT, 100, 0, 1),
            Axis(AxisType.CHEBYSHEV, 100, 0, 1),
        )

        coarse_grid = Grid(
            Axis(AxisType.EQUIDISTANT, 10, 0, 1),
            Axis(AxisType.CHEBYSHEV, 10, 0, 1),
        )

        X, Y = fine_grid.meshed_coords
        f = X**2 + Y**2

        X_c, Y_c = coarse_grid.meshed_coords
        expected = X_c**2 + Y_c**2

        interp = Interpolator(fine_grid, f)

        f_coarse = interp(coarse_grid)

        self.assertEqual(f_coarse.shape, coarse_grid.shape)
        npt.assert_array_almost_equal(f_coarse, expected)

    def test_interpol_grid_extrapol_raises_exception(self):
        fine_grid = Grid(
            Axis(AxisType.EQUIDISTANT, 100, 0, 1),
            Axis(AxisType.CHEBYSHEV, 100, 0, 1),
        )

        coarse_grid = Grid(
            Axis(AxisType.EQUIDISTANT, 10, 0, 2),
            Axis(AxisType.CHEBYSHEV, 10, 0, 2),
        )

        X, Y = fine_grid.meshed_coords
        f = X**2 + Y**2

        interp = Interpolator(fine_grid, f)

        with self.assertRaises(ValueError):
            interp(coarse_grid)
