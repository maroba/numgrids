import unittest

import numpy as np
import numpy.testing as npt

from numgrids.grids import Grid
from numgrids.axes import EquidistantAxis


class TestEquidistantAxis(unittest.TestCase):

    def test_init_axis_with_scaling_and_offset(self):

        axis = EquidistantAxis(11, -3, 7)
        self.assertEqual(11, len(axis))
        self.assertEqual(-3, axis[0])
        self.assertEqual(-2, axis[1])
        self.assertEqual(axis[1], axis.get_coordinate(1))
        npt.assert_array_almost_equal(np.linspace(-3, 7, 11), axis.coords)

        with self.assertRaises(IndexError):
            axis[20]


    def test_init_periodic(self):

        axis = EquidistantAxis(11, -3, 7, periodic=True)
        self.assertEqual(11, len(axis))
        self.assertEqual(-3, axis[0])
        self.assertEqual(-2, axis[1])
        npt.assert_array_almost_equal(np.linspace(-3, 7, 11), axis.coords)

        self.assertEqual(axis[0], axis[11])
        self.assertEqual(axis[1], axis[12])


class TestGridEquidistant(unittest.TestCase):

    def test_init_line_segment(self):
        nx = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        grid = Grid(x_axis)
        x = grid.get_axis().coords
        npt.assert_array_almost_equal(x, np.linspace(-3, 7, nx))

        self.assertAlmostEqual(-3, grid[0])
        self.assertAlmostEqual(-2, grid[1])
        self.assertAlmostEqual(7, grid[-1])

    def test_init_square(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        x = grid.get_axis(0).coords
        y = grid.get_axis(1).coords
        npt.assert_array_almost_equal(x, np.linspace(-3, 7, nx))
        npt.assert_array_almost_equal(y, np.linspace(-3, 7, ny))

        npt.assert_array_almost_equal((-3, -3), grid[0, 0])
        npt.assert_array_almost_equal((-3, 7), grid[0, -1])
        npt.assert_array_almost_equal((7, -2), grid[-1, 1])

    def test_init_circle(self):
        nx = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        grid = Grid(x_axis)
        self.assertEqual(grid[1], grid[12])

    def test_init_torus(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)

        npt.assert_array_almost_equal((-3, -3), grid[0, 0])
        npt.assert_array_almost_equal((-3, 7), grid[0, -1])
        npt.assert_array_almost_equal((7, -2), grid[-1, 1])

        npt.assert_array_almost_equal(grid[0, 0], grid[11, 0])
        npt.assert_array_almost_equal(grid[1, 0], grid[12, 0])

        with self.assertRaises(IndexError):
            npt.assert_array_almost_equal(grid[0, 0], grid[0, 11])

    def test_use_a_grid(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        x, y = grid.coords

        X, Y = grid.meshed_coords

        f = X**2 + Y**2

        npt.assert_array_almost_equal(x_axis.coords, x)
        npt.assert_array_almost_equal(y_axis.coords, y)

        X_, Y_ = np.meshgrid(x, y, indexing="ij")
        npt.assert_array_almost_equal(X, X_)
        npt.assert_array_almost_equal(Y, Y_)

        npt.assert_array_almost_equal(f, X_**2 + Y_**2)
