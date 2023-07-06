import unittest

import numpy as np
import numpy.testing as npt


from numgrids import Axis, AxisType, SphericalGrid, Diff, interpolate, diff, integrate
from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.diff import FiniteDifferenceDiff, FFTDiff, ChebyshevDiff, LogDiff
from numgrids.grids import Grid


class TestDiff(unittest.TestCase):

    def test_equidistant_nonperiodic(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), FiniteDifferenceDiff)

    def test_equidistant_periodic(self):
        axis = EquidistantAxis(100, 0, 1, periodic=True)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), FFTDiff)

    def test_chebyshev_diff(self):
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), ChebyshevDiff)

    def test_log_diff(self):
        axis = LogAxis(20, 0.1, 1)
        grid = Grid(axis, axis)

        d_dx = Diff(grid, 1, 0)

        self.assertEqual(type(d_dx.operator), LogDiff)

    def test_validation(self):
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis, axis)

        with self.assertRaises(ValueError, msg="Negative order must raise exception"):
            Diff(grid, -1, 0)

        with self.assertRaises(ValueError, msg="Negative axis index must raise exception"):
            Diff(grid, 1, -1)

        with self.assertRaises(ValueError, msg="Too high axis index must raise exception"):
            Diff(grid, 1, 20)

    def test_call(self):
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis)
        x = axis.coords
        f = x**2
        d_dx = Diff(grid, 1, 0)
        npt.assert_array_almost_equal(2*x, d_dx(f))

class TestSphericalGrid(unittest.TestCase):

    def test_spherical_grid(self):
        grid = SphericalGrid(
            Axis(AxisType.CHEBYSHEV, 30, 1.E-3, 1),  # radial axis
            Axis(AxisType.CHEBYSHEV, 30, 1.E-3, np.pi - 1.E-3),  # polar axis
            Axis(AxisType.EQUIDISTANT_PERIODIC, 50, 0, 2 * np.pi),  # azimuthal axis
        )
        self.assertTrue(grid.axes[-1].periodic)
        self.assertTrue(len(grid.axes[0]) == 30)


class TestConvenience(unittest.TestCase):

    def test_diff(self):
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis)
        x = axis.coords
        f = x**2

        self.assertEqual(
            0, len(grid.cache["diffs"])
        )

        # With default arguments
        actual =  diff(grid, f)
        npt.assert_array_almost_equal(
            2*x,
            actual
        )

        # With explicit args
        actual =  diff(grid, f, 1, 0)
        npt.assert_array_almost_equal(
            2*x,
            actual
        )

        # Test coverage ensures that the cache has actually been used.
        self.assertEqual(
            1, len(grid.cache["diffs"])
        )

    def test_interpolate(self):
        grid = Grid(Axis(AxisType.EQUIDISTANT, 50, 0, 1))
        x = grid.coords
        f = x ** 2
        expected = 0.5 ** 2

        actual = interpolate(grid, f, 0.5)
        self.assertAlmostEqual(actual, expected)

    def test_integrate(self):
        grid = Grid(
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
            Axis(AxisType.CHEBYSHEV, 30, -1, 1),
        )

        X, Y, Z = grid.meshed_coords
        f = np.sin(X) ** 2 + np.sin(Y) ** 2 + np.sin(Z) ** 2

        expected = -12 * np.sin(1) * np.cos(1) + 12

        self.assertIsNone(grid.cache.get("integral"))

        actual = integrate(grid, f)
        npt.assert_array_almost_equal(actual, expected)

        actual = integrate(grid, f)
        npt.assert_array_almost_equal(actual, expected)

        self.assertIsNotNone(grid.cache.get("integral"))