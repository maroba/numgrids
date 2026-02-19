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

    def test_diff_2d(self):
        r_axis = LogAxis(100, 1.E-3, 100)
        phi_axis = EquidistantAxis(100, 0, 2*np.pi, periodic=True)
        grid = Grid(r_axis, phi_axis)
        R, Phi = grid.meshed_coords
        f = np.cos(Phi) / R

        d_dr = LogDiff(grid, 1, 0)
        actual = d_dr(f)
        expected = - f / R
        error = np.max(np.abs((actual - expected)/expected))

        #plt.plot(x, actual)
        #plt.plot(x, expected)
        #plt.show()

        self.assertTrue(error < 1.E-5, msg=str(error))


class TestAccuracyOrder(unittest.TestCase):
    """Tests for the configurable accuracy order parameter."""

    def test_finite_diff_default_acc(self):
        grid = Grid(EquidistantAxis(100, 0, 1))
        d = FiniteDifferenceDiff(grid, 1, 0)
        self.assertEqual(d.acc, 4)

    def test_finite_diff_custom_acc(self):
        grid = Grid(EquidistantAxis(100, 0, 1))
        d = FiniteDifferenceDiff(grid, 1, 0, acc=2)
        self.assertEqual(d.acc, 2)

    def test_finite_diff_higher_acc_is_more_accurate(self):
        grid = Grid(EquidistantAxis(30, 0, 1))
        x = grid.coords
        f = x ** 5

        d_low = FiniteDifferenceDiff(grid, 1, 0, acc=2)
        d_high = FiniteDifferenceDiff(grid, 1, 0, acc=6)

        expected = 5 * x ** 4
        error_low = np.max(np.abs(d_low(f)[3:-3] - expected[3:-3]))
        error_high = np.max(np.abs(d_high(f)[3:-3] - expected[3:-3]))

        self.assertGreater(error_low, error_high)

    def test_fft_diff_default_acc(self):
        axis = EquidistantAxis(50, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        d = FFTDiff(grid, 1, 0)
        self.assertEqual(d.acc, 6)

    def test_fft_diff_custom_acc(self):
        axis = EquidistantAxis(50, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        d = FFTDiff(grid, 1, 0, acc=2)
        self.assertEqual(d.acc, 2)

    def test_log_diff_default_acc(self):
        grid = Grid(LogAxis(50, 0.1, 10))
        d = LogDiff(grid, 1, 0)
        self.assertEqual(d.acc, 6)

    def test_log_diff_custom_acc(self):
        grid = Grid(LogAxis(50, 0.1, 10))
        d = LogDiff(grid, 1, 0, acc=2)
        self.assertEqual(d.acc, 2)

    def test_log_diff_higher_acc_is_more_accurate(self):
        grid = Grid(LogAxis(50, 0.1, 10))
        x = grid.coords
        f = x ** 3

        d_low = LogDiff(grid, 1, 0, acc=2)
        d_high = LogDiff(grid, 1, 0, acc=6)

        expected = 3 * x ** 2
        error_low = np.max(np.abs(d_low(f)[3:-3] - expected[3:-3]))
        error_high = np.max(np.abs(d_high(f)[3:-3] - expected[3:-3]))

        self.assertGreater(error_low, error_high)


class TestFFTDiffMatrix(unittest.TestCase):
    """Tests for the spectral differentiation matrix of FFTDiff."""

    def test_matrix_matches_operator_1d_order1(self):
        """Matrix-vector product must equal operator output."""
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        f = np.exp(np.sin(x))

        d = FFTDiff(grid, 1, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_matches_operator_1d_order2(self):
        """Matrix must also match operator for second-order derivatives."""
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        f = np.exp(np.sin(x))

        d = FFTDiff(grid, 2, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_matches_operator_2d(self):
        """Matrix must match operator for multi-dimensional grids."""
        axis_r = ChebyshevAxis(20, 0.5, 1)
        axis_theta = EquidistantAxis(32, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis_r, axis_theta)
        R, Theta = grid.meshed_coords
        f = R ** 2 * np.sin(2 * Theta)

        d = FFTDiff(grid, 1, 1)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f.reshape(-1)).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_shape_1d(self):
        n = 50
        axis = EquidistantAxis(n, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        D = FFTDiff(grid, 1, 0).as_matrix()

        self.assertEqual(D.shape, (n, n))

    def test_matrix_shape_2d(self):
        n_r, n_theta = 30, 40
        grid = Grid(
            ChebyshevAxis(n_r, 0.5, 1),
            EquidistantAxis(n_theta, 0, 2 * np.pi, periodic=True)
        )
        D = FFTDiff(grid, 1, 1).as_matrix()
        total = n_r * n_theta

        self.assertEqual(D.shape, (total, total))

    def test_spectral_accuracy_first_derivative(self):
        """Spectral method must achieve near-machine-precision accuracy."""
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        f = np.sin(3 * x)
        expected = 3 * np.cos(3 * x)

        D = FFTDiff(grid, 1, 0).as_matrix()
        result = (D @ f).reshape(grid.shape)

        error = np.max(np.abs(result - expected))
        self.assertLess(error, 1e-12, f"Spectral error {error} is not near machine precision")

    def test_spectral_accuracy_second_derivative(self):
        """Second derivative must also achieve spectral accuracy."""
        axis = EquidistantAxis(64, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        f = np.sin(3 * x)
        expected = -9 * np.sin(3 * x)

        D = FFTDiff(grid, 2, 0).as_matrix()
        result = (D @ f).reshape(grid.shape)

        error = np.max(np.abs(result - expected))
        self.assertLess(error, 1e-11, f"Spectral error {error} is not near machine precision")

    def test_spectral_convergence_is_exponential(self):
        """Error must decay exponentially with grid size (spectral convergence)."""
        errors = []
        for n in [16, 32, 64]:
            axis = EquidistantAxis(n, 0, 2 * np.pi, periodic=True)
            grid = Grid(axis)
            x = axis.coords
            f = np.exp(np.sin(x))
            expected = np.cos(x) * f

            D = FFTDiff(grid, 1, 0).as_matrix()
            result = (D @ f).reshape(grid.shape)
            errors.append(np.max(np.abs(result - expected)))

        # Spectral convergence: error ratio should increase rapidly
        ratio = errors[0] / max(errors[1], 1e-16)
        self.assertGreater(ratio, 100, "Convergence is not exponential")
        self.assertLess(errors[-1], 1e-10, f"Error {errors[-1]} not at spectral level")

    def test_matrix_with_non_2pi_domain(self):
        """Matrix must give correct derivatives for domain period L != 2*pi."""
        L = 4.0
        axis = EquidistantAxis(64, 0, L, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        k = 2 * np.pi / L
        f = np.sin(k * x)
        expected = k * np.cos(k * x)

        D = FFTDiff(grid, 1, 0).as_matrix()
        result = (D @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(result, expected)

    def test_operator_with_non_2pi_domain(self):
        """Operator must give correct derivatives for domain period L != 2*pi."""
        L = 4.0
        axis = EquidistantAxis(64, 0, L, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        k = 2 * np.pi / L
        f = np.sin(k * x)
        expected = k * np.cos(k * x)

        d = FFTDiff(grid, 1, 0)
        result = d(f)

        npt.assert_array_almost_equal(result, expected)

    def test_matrix_odd_grid(self):
        """Matrix must work correctly for odd-sized grids."""
        axis = EquidistantAxis(63, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        x = axis.coords
        f = np.sin(x)
        expected = np.cos(x)

        D = FFTDiff(grid, 1, 0).as_matrix()
        result = (D @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(result, expected)


class TestLogDiffMatrix(unittest.TestCase):
    """Tests for the sparse-matrix representation of LogDiff."""

    def test_matrix_matches_operator_1d(self):
        """Matrix-vector product must equal operator output."""
        axis = LogAxis(100, 1e-3, 2 * np.pi)
        grid = Grid(axis)
        x = grid.coords
        f = np.exp(np.sin(x))

        d = LogDiff(grid, 1, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_matches_operator_1d_order2(self):
        """Matrix must match operator for second-order derivatives."""
        axis = LogAxis(100, 0.1, 10)
        grid = Grid(axis)
        x = grid.coords
        f = x ** 3

        d = LogDiff(grid, 2, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_matches_operator_2d(self):
        """Matrix must match operator for multi-dimensional grids."""
        r_axis = LogAxis(50, 1e-3, 100)
        phi_axis = EquidistantAxis(40, 0, 2 * np.pi, periodic=True)
        grid = Grid(r_axis, phi_axis)
        R, Phi = grid.meshed_coords
        f = np.cos(Phi) / R

        d = LogDiff(grid, 1, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f.ravel()).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)

    def test_matrix_shape_1d(self):
        n = 50
        grid = Grid(LogAxis(n, 0.1, 10))
        D = LogDiff(grid, 1, 0).as_matrix()
        self.assertEqual(D.shape, (n, n))

    def test_matrix_shape_2d(self):
        n_r, n_phi = 30, 40
        grid = Grid(LogAxis(n_r, 0.1, 10), EquidistantAxis(n_phi, 0, 2 * np.pi, periodic=True))
        D = LogDiff(grid, 1, 0).as_matrix()
        total = n_r * n_phi
        self.assertEqual(D.shape, (total, total))

    def test_matrix_accuracy_1d(self):
        """Matrix derivative of x^3 should give 3x^2."""
        axis = LogAxis(200, 0.1, 10)
        grid = Grid(axis)
        x = grid.coords
        f = x ** 3
        expected = 3 * x ** 2

        D = LogDiff(grid, 1, 0).as_matrix()
        result = (D @ f).reshape(grid.shape)

        # Check interior points (boundary stencils are less accurate)
        npt.assert_allclose(result[3:-3], expected[3:-3], rtol=1e-3)

    def test_matrix_2d_non_square_grid(self):
        """Matrix must work when periodic axis is not the last axis."""
        axis_periodic = EquidistantAxis(32, 0, 2 * np.pi, periodic=True)
        axis_other = EquidistantAxis(20, 0, 1)
        grid = Grid(axis_periodic, axis_other)
        X, Y = grid.meshed_coords
        f = np.sin(X) * Y ** 2

        d = FFTDiff(grid, 1, 0)
        expected = d(f)
        matrix_result = (d.as_matrix() @ f.reshape(-1)).reshape(grid.shape)

        npt.assert_array_almost_equal(matrix_result, expected)
