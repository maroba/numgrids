import unittest
from unittest.mock import patch, Mock

import warnings

import numpy as np
import numpy.testing as npt


from numgrids import create_axis, AxisType, SphericalGrid, CylindricalGrid, Diff, interpolate, diff, integrate
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

    def test_default_acc(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis)
        d = Diff(grid, 1, 0)
        self.assertEqual(d.operator.acc, 4)

    def test_custom_acc(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis)
        d = Diff(grid, 1, 0, acc=8)
        self.assertEqual(d.operator.acc, 8)

    def test_acc_ignored_by_chebyshev(self):
        axis = ChebyshevAxis(20, 0, 1)
        grid = Grid(axis)
        x = axis.coords
        f = x ** 2
        # acc is accepted but ignored — result should still be accurate
        d = Diff(grid, 1, 0, acc=2)
        npt.assert_array_almost_equal(2 * x, d(f))

    def test_acc_with_periodic_axis(self):
        axis = EquidistantAxis(50, 0, 2 * np.pi, periodic=True)
        grid = Grid(axis)
        d = Diff(grid, 1, 0, acc=2)
        self.assertEqual(d.operator.acc, 2)

    def test_acc_with_log_axis(self):
        axis = LogAxis(50, 0.1, 10)
        grid = Grid(axis)
        d = Diff(grid, 1, 0, acc=8)
        self.assertEqual(d.operator.acc, 8)

class TestSphericalGrid(unittest.TestCase):

    def test_spherical_grid(self):
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 30, 1.E-3, 1),  # radial axis
            create_axis(AxisType.CHEBYSHEV, 30, 1.E-3, np.pi - 1.E-3),  # polar axis
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 50, 0, 2 * np.pi),  # azimuthal axis
        )
        self.assertTrue(grid.axes[-1].periodic)
        self.assertTrue(len(grid.axes[0]) == 30)

    def test_spherical_grid_lazy_laplacian(self):
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 15, 1.E-3, 1),
            create_axis(AxisType.CHEBYSHEV, 15, 1.E-3, np.pi - 1.E-3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        # Laplacian is not initialized until called
        self.assertIsNone(grid._laplacian_fn)
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2
        # Calling laplacian triggers lazy init
        result = grid.laplacian(f)
        self.assertIsNotNone(grid._laplacian_fn)
        self.assertEqual(result.shape, f.shape)

    def test_no_warnings_away_from_singularity(self):
        """Grid far from singularities should produce no warnings."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 15, 0.5, 2),
            create_axis(AxisType.CHEBYSHEV, 15, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = grid.laplacian(f)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_no_warnings_near_r_zero(self):
        """Grid near r=0 must not produce RuntimeWarnings."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 20, 1.E-6, 1),
            create_axis(AxisType.CHEBYSHEV, 15, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = grid.laplacian(f)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_no_warnings_near_theta_zero(self):
        """Grid near theta=0 must not produce RuntimeWarnings."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 15, 0.5, 2),
            create_axis(AxisType.CHEBYSHEV, 15, 1.E-6, np.pi - 1.E-6),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = grid.laplacian(f)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_no_nan_or_inf_at_singularities(self):
        """Output must be finite even when grid includes near-singular points."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 20, 1.E-8, 1),
            create_axis(AxisType.CHEBYSHEV, 20, 1.E-8, np.pi - 1.E-8),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2 * np.sin(Theta) ** 2

        result = grid.laplacian(f)

        self.assertFalse(np.any(np.isnan(result)), "NaN found in laplacian output")
        self.assertFalse(np.any(np.isinf(result)), "Inf found in laplacian output")

    def test_laplacian_r_squared(self):
        """Laplacian of r^2 = 6 in spherical coordinates."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 25, 0.1, 2),
            create_axis(AxisType.CHEBYSHEV, 20, 0.2, np.pi - 0.2),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        result = grid.laplacian(f)
        expected = 6 * np.ones_like(f)

        # Check at interior points away from boundaries
        interior = (
            (slice(2, -2), slice(2, -2), slice(None))
        )
        npt.assert_array_almost_equal(result[interior], expected[interior], decimal=1)

    def test_laplacian_1_over_r(self):
        """Laplacian of 1/r = 0 for r > 0 (harmonic function)."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 25, 0.5, 3),
            create_axis(AxisType.CHEBYSHEV, 20, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 20, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = 1.0 / R

        result = grid.laplacian(f)

        interior = (slice(2, -2), slice(2, -2), slice(None))
        npt.assert_array_almost_equal(result[interior], 0.0, decimal=1)

    def test_laplacian_uses_correct_theta_phi_ordering(self):
        """Verify theta and phi are not swapped (regression test)."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 20, 0.5, 2),
            create_axis(AxisType.CHEBYSHEV, 20, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords

        # f = cos(phi) depends only on azimuthal angle
        # Laplacian should have non-zero phi terms but zero theta-only terms
        f = np.cos(Phi)

        result = grid.laplacian(f)

        # With correct ordering, the phi derivative term contributes.
        # If theta/phi were swapped, result would be very different.
        # At interior points away from boundaries, check that the result
        # matches the analytical form: -cos(phi) / (r^2 sin^2(theta))
        interior = (slice(3, -3), slice(3, -3), slice(None))
        expected = -np.cos(Phi) / (R ** 2 * np.sin(Theta) ** 2)
        error = np.max(np.abs(result[interior] - expected[interior]))
        self.assertLess(error, 1.0, f"Theta/phi ordering appears wrong, error={error}")

    def test_laplacian_second_call_uses_cache(self):
        """Second call to laplacian should reuse the cached function."""
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 10, 0.5, 1),
            create_axis(AxisType.CHEBYSHEV, 10, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2

        result1 = grid.laplacian(f)
        fn_ref = grid._laplacian_fn
        result2 = grid.laplacian(f)

        self.assertIs(grid._laplacian_fn, fn_ref)
        npt.assert_array_equal(result1, result2)


class TestCylindricalGrid(unittest.TestCase):

    def _make_grid(self, r_low=0.1, r_high=2, nr=20, nphi=30, nz=20,
                   z_low=-1, z_high=1):
        """Helper to create a standard cylindrical grid."""
        return CylindricalGrid(
            create_axis(AxisType.CHEBYSHEV, nr, r_low, r_high),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, nphi, 0, 2 * np.pi),
            create_axis(AxisType.CHEBYSHEV, nz, z_low, z_high),
        )

    def test_construction(self):
        grid = self._make_grid()
        self.assertEqual(grid.ndims, 3)
        self.assertEqual(len(grid.axes), 3)
        self.assertTrue(grid.axes[1].periodic)

    def test_shape(self):
        grid = self._make_grid(nr=15, nphi=20, nz=10)
        self.assertEqual(grid.shape, (15, 20, 10))

    def test_meshed_coords_ordering(self):
        """Axes order must be (r, phi, z)."""
        grid = self._make_grid()
        R, Phi, Z = grid.meshed_coords
        # R should increase along axis 0
        self.assertTrue(np.all(np.diff(R[:, 0, 0]) > 0) or
                        np.all(np.diff(R[:, 0, 0]) < 0))
        # Z should increase along axis 2
        self.assertTrue(np.all(np.diff(Z[0, 0, :]) > 0) or
                        np.all(np.diff(Z[0, 0, :]) < 0))

    def test_lazy_laplacian_init(self):
        """Laplacian should not be set up until first call."""
        grid = self._make_grid()
        self.assertIsNone(grid._laplacian_fn)
        R, Phi, Z = grid.meshed_coords
        grid.laplacian(R ** 2)
        self.assertIsNotNone(grid._laplacian_fn)

    def test_laplacian_caching(self):
        """Second call must reuse the cached Laplacian function."""
        grid = self._make_grid()
        R, Phi, Z = grid.meshed_coords
        f = R ** 2
        grid.laplacian(f)
        fn_ref = grid._laplacian_fn
        grid.laplacian(f)
        self.assertIs(grid._laplacian_fn, fn_ref)

    def test_laplacian_output_shape(self):
        grid = self._make_grid()
        R, Phi, Z = grid.meshed_coords
        f = R ** 2 + Z ** 2
        result = grid.laplacian(f)
        self.assertEqual(result.shape, grid.shape)

    def test_laplacian_r_squared_plus_z_squared(self):
        r"""Laplacian of r^2 + z^2.

        d^2/dr^2(r^2) + (1/r)*d/dr(r^2) + d^2/dz^2(z^2)
        = 2 + (1/r)*2r + 2 = 6
        """
        grid = self._make_grid(nr=25, nphi=20, nz=25)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2 + Z ** 2
        result = grid.laplacian(f)
        expected = 6 * np.ones_like(f)

        interior = (slice(2, -2), slice(None), slice(2, -2))
        npt.assert_array_almost_equal(result[interior], expected[interior], decimal=1)

    def test_laplacian_ln_r(self):
        r"""Laplacian of ln(r) = 0 for r > 0 (harmonic in 2D cylindrical).

        d^2/dr^2(ln r) + (1/r) d/dr(ln r) = -1/r^2 + 1/r^2 = 0
        """
        grid = self._make_grid(r_low=0.5, r_high=3, nr=30, nphi=20, nz=15)
        R, Phi, Z = grid.meshed_coords
        f = np.log(R)
        result = grid.laplacian(f)

        interior = (slice(3, -3), slice(None), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 0.0, decimal=1)

    def test_laplacian_cos_phi_over_r(self):
        r"""Laplacian of cos(phi)/r = 0 for r > 0 (harmonic function).

        This function is harmonic: nabla^2(cos(phi)/r) = 0.
        """
        grid = self._make_grid(r_low=0.5, r_high=3, nr=25, nphi=40, nz=15)
        R, Phi, Z = grid.meshed_coords
        f = np.cos(Phi) / R
        result = grid.laplacian(f)

        interior = (slice(3, -3), slice(None), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 0.0, decimal=1)

    def test_laplacian_z_squared(self):
        r"""Laplacian of z^2 = 2 (no r or phi dependence)."""
        grid = self._make_grid(nr=15, nphi=20, nz=25)
        R, Phi, Z = grid.meshed_coords
        f = Z ** 2
        result = grid.laplacian(f)
        expected = 2 * np.ones_like(f)

        interior = (slice(2, -2), slice(None), slice(2, -2))
        npt.assert_array_almost_equal(result[interior], expected[interior], decimal=2)

    def test_laplacian_r_squared(self):
        r"""Laplacian of r^2 = 4 (only radial terms contribute).

        d^2/dr^2(r^2) + (1/r)*d/dr(r^2) = 2 + 2 = 4
        """
        grid = self._make_grid(nr=25, nphi=20, nz=15)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2
        result = grid.laplacian(f)
        expected = 4 * np.ones_like(f)

        interior = (slice(2, -2), slice(None), slice(2, -2))
        npt.assert_array_almost_equal(result[interior], expected[interior], decimal=1)

    def test_no_warnings_away_from_singularity(self):
        """No RuntimeWarnings when grid is far from r=0."""
        grid = self._make_grid(r_low=0.5, r_high=2)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = grid.laplacian(f)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_no_warnings_near_r_zero(self):
        """No RuntimeWarnings even when r is very close to zero."""
        grid = self._make_grid(r_low=1e-6, r_high=1)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = grid.laplacian(f)

        self.assertTrue(np.all(np.isfinite(result)))

    def test_no_nan_or_inf_at_singularity(self):
        """Output must be finite even with grid very close to r=0."""
        grid = self._make_grid(r_low=1e-10, r_high=1)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2 * np.cos(Phi)
        result = grid.laplacian(f)

        self.assertFalse(np.any(np.isnan(result)), "NaN in output")
        self.assertFalse(np.any(np.isinf(result)), "Inf in output")

    def test_phi_dependence(self):
        r"""Laplacian of r^2 * cos(2*phi) must include azimuthal term.

        nabla^2(r^2 cos(2phi)) = 4*cos(2phi) - 4*cos(2phi) = 0
        (The r terms give 4*cos(2phi), the phi term gives -4*cos(2phi))
        """
        grid = self._make_grid(r_low=0.5, r_high=3, nr=25, nphi=40, nz=15)
        R, Phi, Z = grid.meshed_coords
        f = R ** 2 * np.cos(2 * Phi)
        result = grid.laplacian(f)

        interior = (slice(3, -3), slice(None), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 0.0, decimal=1)

    def test_is_subclass_of_grid(self):
        grid = self._make_grid()
        self.assertIsInstance(grid, Grid)


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

    def test_diff_with_acc(self):
        axis = EquidistantAxis(100, 0, 1)
        grid = Grid(axis)
        x = axis.coords
        f = x ** 4

        result = diff(grid, f, 1, 0, acc=6)
        npt.assert_array_almost_equal(result, 4 * x ** 3, decimal=5)

        # Different acc values create separate cache entries
        diff(grid, f, 1, 0, acc=2)
        self.assertEqual(2, len(grid.cache["diffs"]))
        self.assertIn((1, 0, 6), grid.cache["diffs"])
        self.assertIn((1, 0, 2), grid.cache["diffs"])

    def test_interpolate(self):
        grid = Grid(create_axis(AxisType.EQUIDISTANT, 50, 0, 1))
        x = grid.coords
        f = x ** 2
        expected = 0.5 ** 2

        actual = interpolate(grid, f, 0.5)
        self.assertAlmostEqual(actual, expected)

    def test_create_axis_logarithmic(self):
        axis = create_axis(AxisType.LOGARITHMIC, 20, 0.1, 10)
        self.assertIsInstance(axis, LogAxis)
        self.assertEqual(len(axis), 20)

    def test_create_axis_invalid_type(self):
        with self.assertRaises(NotImplementedError):
            create_axis("nonexistent", 10, 0, 1)

    def test_type_error_for_non_grid(self):
        with self.assertRaises(TypeError):
            Diff("not a grid", 1, 0)

    def test_grid_plot(self):
        axis = EquidistantAxis(10, 0, 1)
        grid = Grid(axis, axis)
        with patch("matplotlib.pyplot.figure") as mock_fig, \
             patch("matplotlib.pyplot.show"):
            mock_fig.return_value = Mock()
            mock_fig.return_value.add_subplot = Mock(return_value=Mock())
            grid.plot()
            mock_fig.assert_called()

    def test_grid_plot_with_named_axes(self):
        axis_x = EquidistantAxis(10, 0, 1, name="x")
        axis_y = EquidistantAxis(10, 0, 1, name="y")
        grid = Grid(axis_x, axis_y)
        with patch("matplotlib.pyplot.figure") as mock_fig, \
             patch("matplotlib.pyplot.show"):
            mock_sub = Mock()
            mock_fig.return_value = Mock()
            mock_fig.return_value.add_subplot = Mock(return_value=mock_sub)
            grid.plot()
            mock_sub.set_xlabel.assert_called()
            mock_sub.set_ylabel.assert_called()

    def test_integrate(self):
        grid = Grid(
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
            create_axis(AxisType.CHEBYSHEV, 30, -1, 1),
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