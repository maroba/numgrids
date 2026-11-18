import unittest

import numpy as np
import numpy.testing as npt
import pytest

from numgrids import AxisType, Grid, create_axis
from numgrids.integration import Integral


class TestIntegral(unittest.TestCase):

    def test_integral_1d_findiff(self):
        grid = Grid(
            create_axis(AxisType.EQUIDISTANT, 100, -1, 1)
        )

        x = grid.coords
        f = np.cos(x)

        expected = np.sin(1) - np.sin(-1)

        I = Integral(grid)

        actual = I(f)

        npt.assert_array_almost_equal(actual, expected)

    def test_integral_2d_findiff(self):
        grid = Grid(
            create_axis(AxisType.EQUIDISTANT, 100, -1, 1),
            create_axis(AxisType.CHEBYSHEV, 100, 0, 1)
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
            create_axis(AxisType.EQUIDISTANT, 50, -1, 1),
            create_axis(AxisType.EQUIDISTANT, 50, -1, 1),
            create_axis(AxisType.EQUIDISTANT, 50, -1, 1),
        )

        X, Y, Z = grid.meshed_coords
        f = np.sin(X) ** 2 + np.sin(Y) ** 2 + np.sin(Z) ** 2

        I = Integral(grid)

        expected = -12 * np.sin(1) * np.cos(1) + 12

        actual = I(f)
        npt.assert_array_almost_equal(actual, expected, decimal=5)

    def test_integral_3d_findiff_cheb(self):
        grid = Grid(
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
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
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 4*np.pi),
        )

        x = grid.coords
        f = np.cos(x)**2

        I = Integral(grid)

        expected = 2*np.pi
        actual = I(f)

        self.assertAlmostEqual(actual, expected)


class TestInnerProductsAndNorms:
    def grids(self):
        from numgrids import PolarGrid, SphericalGrid, chebyshev, colatitude, equidistant, logarithmic, periodic, radial
        return [
            Grid(x=chebyshev(12, 0, 2), y=periodic(8)),
            Grid(x=equidistant(11, 0, 1)),
            Grid(logarithmic(12, 0.1, 2)),
            PolarGrid(radial(8, 1.0), periodic(16)),
            SphericalGrid(radial(6, 1.0), colatitude(6), periodic(12)),
        ]

    def test_quadrature_weights_reproduce_integrate(self):
        for grid in self.grids():
            f = grid.evaluate(lambda *c: np.exp(c[0]))
            W = grid.quadrature_weights
            assert W.shape == grid.shape
            assert np.sum(W * f) == pytest.approx(grid.integrate(f), rel=1e-13)
            assert not W.flags.writeable

    def test_volumes(self):
        disk, ball = self.grids()[3:]
        assert disk.quadrature_weights.sum() == pytest.approx(np.pi, rel=1e-12)
        assert ball.quadrature_weights.sum() == pytest.approx(4 * np.pi / 3, rel=1e-12)

    def test_inner_product(self):
        from numgrids import periodic
        grid = Grid(x=periodic(16))
        x = grid.coords
        assert grid.inner(np.sin(x), np.sin(x)) == pytest.approx(np.pi)
        assert grid.inner(np.sin(x), np.cos(x)) == pytest.approx(0.0, abs=1e-12)
        # conjugate-linear in the first argument
        f, g = np.exp(1j * x), np.exp(2j * x) + 3 * np.exp(1j * x)
        assert grid.inner(f, g) == pytest.approx(6 * np.pi)
        assert grid.inner(1j * f, g) == pytest.approx(-6j * np.pi)
        with pytest.raises(ValueError):
            grid.inner(f, g[:-1])

    def test_norms_and_mean(self):
        from numgrids import chebyshev
        grid = Grid(x=chebyshev(20, 0, 2))
        x = grid.coords
        assert grid.norm(np.ones(grid.shape)) == pytest.approx(np.sqrt(2))
        assert grid.norm(x - 1, 1) == pytest.approx(1.0, abs=1e-2)          # |x - 1| has a kink
        assert grid.norm(-3 * x, "max") == 6.0
        assert grid.norm(1j * x, np.inf) == 2.0
        assert grid.mean(x ** 2) == pytest.approx(4 / 3)
        with pytest.raises(ValueError, match="norm"):
            grid.norm(x, 3)

    def test_mean_on_the_ball(self):
        ball = self.grids()[-1]
        r = ball.meshed_coords[0]
        assert ball.mean(r ** 2) == pytest.approx(3 / 5, rel=1e-10)
