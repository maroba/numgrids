import unittest

import numpy as np
import numpy.testing as npt

from numgrids.axes import EquidistantAxis
from numgrids.diff import FiniteDifferenceDiff
from numgrids.grids import Grid


class TestEquidistantGridDiff(unittest.TestCase):

    def test_1d_diff(self):

        grid = Grid(EquidistantAxis(100, 0, 1))
        x = grid.get_axis().coords
        f = x**4

        d_dx = FiniteDifferenceDiff(grid=grid, order=1, axis_index=0)
        df_fx = d_dx(f)
        npt.assert_array_almost_equal(df_fx, 4*x**3, decimal=7)

        d2_dx2 = FiniteDifferenceDiff(grid=grid, order=2, axis_index=0)
        d2f_fx2 = d2_dx2(f)
        npt.assert_array_almost_equal(d2f_fx2, 12*x**2, decimal=7)

    def test_2d_diff(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis, axis)

        X, Y = grid.meshed_coords
        f = X**3  + Y**3
        d2_dx2 = FiniteDifferenceDiff(grid, 2, 0)
        d2_dy2 = FiniteDifferenceDiff(grid, 2, 1)
        laplace = lambda f: d2_dx2(f) + d2_dy2(f)

        npt.assert_array_almost_equal(6*(X+Y), laplace(f))




