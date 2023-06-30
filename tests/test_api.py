import unittest

import numpy as np
import numpy.testing as npt


from numgrids import Axis, AxisType, SphericalGrid, Diff
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
            Axis.of_type(AxisType.CHEBYSHEV, 30, 1.E-3, 1),  # radial axis
            Axis.of_type(AxisType.CHEBYSHEV, 30, 1.E-3, np.pi - 1.E-3),  # polar axis
            Axis.of_type(AxisType.EQUIDISTANT_PERIODIC, 50, 0, 2 * np.pi),  # azimuthal axis
        )
        self.assertTrue(grid.axes[-1].periodic)
        self.assertTrue(len(grid.axes[0]) == 30)
