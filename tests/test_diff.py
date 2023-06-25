import unittest
# import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
from matplotlib import pyplot as plt

from numgrids.axes import EquidistantAxis, ChebyshevAxis, LogAxis
from numgrids.diff import FiniteDifferenceDiff, FFTDiff, ChebyshevDiff, LogDiff
from numgrids.grids import Grid

np.set_printoptions(linewidth=120)


class TestEquidistantGridDiff(unittest.TestCase):

    def test_1d_diff(self):
        grid = Grid(EquidistantAxis(100, 0, 1))
        x = grid.get_axis().coords
        f = x ** 4

        d_dx = FiniteDifferenceDiff(grid=grid, order=1, axis_index=0)
        df_fx = d_dx(f)
        npt.assert_array_almost_equal(df_fx, 4 * x ** 3, decimal=7)

        d2_dx2 = FiniteDifferenceDiff(grid=grid, order=2, axis_index=0)
        d2f_fx2 = d2_dx2(f)
        npt.assert_array_almost_equal(d2f_fx2, 12 * x ** 2, decimal=7)

    def test_2d_diff(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis, axis)

        X, Y = grid.meshed_coords
        f = X ** 3 + Y ** 3
        d2_dx2 = FiniteDifferenceDiff(grid, 2, 0)
        d2_dy2 = FiniteDifferenceDiff(grid, 2, 1)
        laplace = lambda f: d2_dx2(f) + d2_dy2(f)

        npt.assert_array_almost_equal(6 * (X + Y), laplace(f))


class TestFFTDiff(unittest.TestCase):

    def test_fftdiff_1d_order_1_even_grid(self):
        axis = EquidistantAxis(100, 0, 2 * np.pi, periodic=True)
        x = axis.coords
        f = np.exp(np.sin(x))
        grid = Grid(axis)

        d_dx = FFTDiff(grid, 1, 0)

        actual = d_dx(f)
        expected = np.cos(x) * f

        #        plt.plot(x, expected, "r-")
        #        plt.plot(x, actual)
        #        plt.show()

        npt.assert_array_almost_equal(actual, expected)

    def test_fftdiff_1d_order_2_even_grid(self):
        axis = EquidistantAxis(100, 0, 2 * np.pi, periodic=True)
        x = axis.coords
        f = np.exp(np.sin(x))
        grid = Grid(axis)

        d2_dx2 = FFTDiff(grid, 2, 0)

        actual = d2_dx2(f)
        expected = np.cos(x) ** 2 * f - np.sin(x) * f

        #        plt.plot(x, expected, "r-")
        #        plt.plot(x, actual)
        #        plt.show()

        npt.assert_array_almost_equal(actual, expected)

    def test_fftdiff_1d_order_1_odd_grid(self):
        axis = EquidistantAxis(21, 0, 2 * np.pi, periodic=True)
        x = axis.coords
        f = np.exp(np.sin(x))
        grid = Grid(axis)

        d_dx = FFTDiff(grid, 1, 0)

        actual = d_dx(f)
        expected = np.cos(x) * f

        #        plt.plot(x, expected, "r-")
        #        plt.plot(x, actual)
        #        plt.show()

        npt.assert_array_almost_equal(actual, expected)

    def test_fftdiff_1d_order_2_odd_grid(self):
        axis = EquidistantAxis(21, 0, 2 * np.pi, periodic=True)
        x = axis.coords
        f = np.exp(np.sin(x))
        grid = Grid(axis)

        d2_dx2 = FFTDiff(grid, 2, 0)

        actual = d2_dx2(f)
        expected = np.cos(x) ** 2 * f - np.sin(x) * f

        #        plt.plot(x, expected, "r-")
        #        plt.plot(x, actual)
        #        plt.show()

        npt.assert_array_almost_equal(actual, expected)

    def test_fftdiff_2d_order_1_even_grid(self):
        axis = EquidistantAxis(30, 0, 2 * np.pi, periodic=True)

        grid = Grid(axis, axis)
        X, Y = grid.meshed_coords
        f = np.exp(np.sin(X))

        d_dx = FFTDiff(grid, 1, 0)

        actual = d_dx(f)
        expected = np.cos(X) * f

        #        plt.plot(x, expected, "r-")
        #        plt.plot(x, actual)
        #        plt.show()

        npt.assert_array_almost_equal(actual, expected)

        d_dy = FFTDiff(grid, 1, 1)
        actual = d_dy(f)
        expected = np.zeros_like(f)
        npt.assert_array_almost_equal(actual, expected)

        f = np.exp(np.sin(Y))
        actual = d_dy(f)
        expected = np.cos(Y) * f
        npt.assert_array_almost_equal(actual, expected)


class TestChebyshevDiff(unittest.TestCase):

    def test_diff_1d(self):
        grid = Grid(ChebyshevAxis(21, -1, 1))
        x = grid.coords
        f = np.exp(x) * np.sin(5 * x)

        d_dx = ChebyshevDiff(grid, 1, 0)
        actual = d_dx(f)
        expected = f + 5 * np.cos(5 * x) * np.exp(x)

        npt.assert_array_almost_equal(actual, expected)

    def test_diff_1d_order_2(self):
        grid = Grid(ChebyshevAxis(21, -1, 1))
        x = grid.coords
        f = np.exp(x) * np.sin(5 * x)

        d2_dx2 = ChebyshevDiff(grid, 2, 0)
        actual = d2_dx2(f)
        expected = 2 * np.exp(x) * (5 * np.cos(5 * x) - 12 * np.sin(5 * x))

        npt.assert_array_almost_equal(actual, expected)

    def test_diff_1d_shifted(self):
        grid = Grid(ChebyshevAxis(21, 0, 2))
        x = grid.coords
        f = np.exp(x) * np.sin(5 * x)

        d_dx = ChebyshevDiff(grid, 1, 0)
        actual = d_dx(f)
        expected = f + 5 * np.cos(5 * x) * np.exp(x)

        npt.assert_array_almost_equal(actual, expected)

    def test_diff_1d_scaled(self):
        grid = Grid(ChebyshevAxis(25, -2, 2))
        x = grid.coords
        f = np.exp(x) * np.sin(5 * x)

        d_dx = ChebyshevDiff(grid, 1, 0)
        actual = d_dx(f)
        expected = f + 5 * np.cos(5 * x) * np.exp(x)

        npt.assert_array_almost_equal(actual, expected)

    def test_diff_2d_df_dx(self):
        axis = ChebyshevAxis(25, -1, 1)
        grid = Grid(axis, axis)
        X, Y = grid.meshed_coords

        f = np.exp(X) * np.sin(5 * X)
        d_dx = ChebyshevDiff(grid, 1, 0)
        actual = d_dx(f)
        expected = f + 5 * np.cos(5 * X) * np.exp(X)
        npt.assert_array_almost_equal(actual, expected)

    def test_diff_2d_df_dy(self):
        axis = ChebyshevAxis(25, -1, 1)
        grid = Grid(axis, axis)
        X, Y = grid.meshed_coords

        f = np.exp(Y) * np.sin(5 * Y)
        d_dy = ChebyshevDiff(grid, 1, 1)
        actual = d_dy(f)
        expected = f + 5 * np.cos(5 * Y) * np.exp(Y)
        npt.assert_array_almost_equal(actual, expected)

    def test_diff_3d_df_dy(self):
        axis = ChebyshevAxis(25, -1, 1)
        grid = Grid(axis, axis, axis)
        X, Y, Z = grid.meshed_coords

        f = np.exp(Y) * np.sin(5 * Y)
        d_dy = ChebyshevDiff(grid, 1, 1)
        actual = d_dy(f)
        expected = f + 5 * np.cos(5 * Y) * np.exp(Y)
        npt.assert_array_almost_equal(actual, expected)

    def test_diff_3d_df_dz(self):
        axis = ChebyshevAxis(25, -1, 1)
        grid = Grid(axis, axis, axis)
        X, Y, Z = grid.meshed_coords

        f = np.exp(Z) * np.sin(5 * Z)
        d_dy = ChebyshevDiff(grid, 1, 2)
        actual = d_dy(f)
        expected = f + 5 * np.cos(5 * Z) * np.exp(Z)
        npt.assert_array_almost_equal(actual, expected)

    def test_diff_3d_df_dz_scaled_shifted(self):
        axis = ChebyshevAxis(25, 0, 1)
        grid = Grid(axis, axis, axis)
        X, Y, Z = grid.meshed_coords

        f = np.exp(Z) * np.sin(5 * Z)
        d_dy = ChebyshevDiff(grid, 1, 2)
        actual = d_dy(f)
        expected = f + 5 * np.cos(5 * Z) * np.exp(Z)
        npt.assert_array_almost_equal(actual, expected)

    def test_diff_matrix_1d(self):
        grid = Grid(ChebyshevAxis(3, -1, 1))

        d_dx = ChebyshevDiff(grid, 1, 0)

        expected = - np.array([[1.5, -2, 0.5],
                               [0.5, 0, -0.5],
                               [-0.5, 2, -1.5]])

        npt.assert_array_almost_equal(d_dx.as_matrix().toarray(),
                                      expected
                                      )

    def test_diff_matrix_2d_x(self):
        axis = ChebyshevAxis(3, -1, 1)
        grid = Grid(axis, axis)

        d_dx = ChebyshevDiff(grid, 1, 0)

        D = - np.array([[1.5, -2, 0.5],
                        [0.5, 0, -0.5],
                        [-0.5, 2, -1.5]])

        expected = np.kron(D, np.eye(3))
        npt.assert_array_almost_equal(d_dx.as_matrix().toarray(),
                                      expected
                                      )

    def test_diff_matrix_2d_y(self):
        axis = ChebyshevAxis(3, -1, 1)
        grid = Grid(axis, axis)

        d_dx = ChebyshevDiff(grid, 1, 1)

        D = - np.array([[1.5, -2, 0.5],
                        [0.5, 0, -0.5],
                        [-0.5, 2, -1.5]])

        expected = np.kron(np.eye(3), D)
        npt.assert_array_almost_equal(d_dx.as_matrix().toarray(),
                                      expected
                                      )


class TestLogDiff(unittest.TestCase):

    def test_diff_1d(self):
        axis = LogAxis(300, 1.E-3, 2*np.pi)
        grid = Grid(axis)
        x = grid.coords
        f = np.exp(np.sin(x))

        d_dx = LogDiff(grid, 1, 0)
        actual = d_dx(f)
        expected = np.cos(x)*f
        error = np.max(np.abs((actual - expected)/expected))

        #plt.plot(x, actual)
        #plt.plot(x, expected)
        #plt.show()

        self.assertTrue(error < 1.E-3, msg=str(error))

