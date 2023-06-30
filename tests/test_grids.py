import unittest

from unittest.mock import patch
import numpy as np
import numpy.testing as npt

from numgrids.grids import Grid
from numgrids.axes import EquidistantAxis, ChebyshevAxis


class TestEquidistantAxis(unittest.TestCase):

    def test_init_axis_with_scaling_and_offset(self):

        axis = EquidistantAxis(11, -3, 7)
        self.assertEqual(11, len(axis))
        self.assertEqual(-3, axis[0])
        self.assertEqual(-2, axis[1])
        npt.assert_array_almost_equal(np.linspace(-3, 7, 11), axis.coords)

        with self.assertRaises(IndexError):
            axis[20]


    def test_init_periodic(self):
        axis = EquidistantAxis(10, 0, 1, periodic=True)
        self.assertEqual(10, len(axis))
        self.assertEqual(0, axis[0])
        self.assertEqual(0.9, axis[-1])
        npt.assert_array_almost_equal(np.linspace(0, 1, 10, endpoint=False), axis.coords)

        self.assertEqual(axis[0], axis[10])
        self.assertEqual(axis[1], axis[11])


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

    def test_use_meshed(self):
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

    def test_boundary(self):
        nx = ny = 11
        x_axis = EquidistantAxis(nx, -3, 7)
        y_axis = EquidistantAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        X, Y = grid.meshed_coords

        bdry = np.ones_like(X, dtype=bool)
        bdry[1:-1, 1:-1] = False

        npt.assert_array_equal(grid.boundary, bdry)

    def test_boundary_cheb(self):
        nx = ny = 11
        x_axis = ChebyshevAxis(nx, -3, 7)
        y_axis = ChebyshevAxis(ny, -3, 7)
        grid = Grid(x_axis, y_axis)
        X, Y = grid.meshed_coords

        bdry = np.ones_like(X, dtype=bool)
        bdry[1:-1, 1:-1] = False

        npt.assert_array_equal(grid.boundary, bdry)

    def test_boundary_periodic(self):

        phi_axis = EquidistantAxis(11, 0, 2*np.pi, periodic=True)
        r_axis = EquidistantAxis(11, 1.E-3, 1)
        grid = Grid(r_axis, phi_axis)
        R, Phi = grid.meshed_coords

        bdry = np.ones_like(R, dtype=bool)
        bdry[1:-1, :] = False

        npt.assert_array_equal(grid.boundary, bdry)



    def test_repr(self):
        with patch("matplotlib.pyplot.figure") as figure_mock:
            nx = 11
            x_axis = EquidistantAxis(nx, -3, 7, periodic=True)
            y_axis = EquidistantAxis(nx, -3, 7, periodic=True)
            grid = Grid(x_axis, y_axis)
            repr(grid)
            figure_mock.assert_called()