import unittest

import numpy as np
import numpy.testing as npt

from numgrids import (
    CurvilinearGrid, create_axis, AxisType,
    SphericalGrid, CylindricalGrid, PolarGrid,
    Grid,
)
from numgrids.axes import EquidistantAxis, ChebyshevAxis


class TestCurvilinearGridConstruction(unittest.TestCase):
    """Basic construction and validation tests."""

    def test_wrong_number_of_scale_factors_raises(self):
        ax = ChebyshevAxis(10, 0, 1)
        with self.assertRaises(ValueError):
            CurvilinearGrid(ax, scale_factors=(lambda c: c[0], lambda c: c[0]))

    def test_construction_1d(self):
        ax = ChebyshevAxis(20, 0, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        self.assertEqual(grid.ndims, 1)
        self.assertEqual(grid.shape, (20,))

    def test_construction_2d(self):
        ax0 = ChebyshevAxis(15, 0, 1)
        ax1 = EquidistantAxis(20, 0, 2 * np.pi, periodic=True)
        grid = CurvilinearGrid(
            ax0, ax1,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: c[0],
            ),
        )
        self.assertEqual(grid.ndims, 2)
        self.assertEqual(grid.shape, (15, 20))

    def test_construction_3d(self):
        ax0 = ChebyshevAxis(10, 0.1, 1)
        ax1 = EquidistantAxis(15, 0, 2 * np.pi, periodic=True)
        ax2 = ChebyshevAxis(10, -1, 1)
        grid = CurvilinearGrid(
            ax0, ax1, ax2,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: c[0],
                lambda c: np.ones_like(c[0]),
            ),
        )
        self.assertEqual(grid.ndims, 3)

    def test_is_subclass_of_grid(self):
        ax = ChebyshevAxis(10, 0, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        self.assertIsInstance(grid, Grid)

    def test_inherits_grid_properties(self):
        """CurvilinearGrid should have all standard Grid properties."""
        ax = ChebyshevAxis(10, 0, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        self.assertIsNotNone(grid.coords)
        self.assertIsNotNone(grid.boundary)
        self.assertIsNotNone(grid.meshed_coords)


class TestCurvilinearGridCartesian(unittest.TestCase):
    """Cartesian coordinates: all scale factors = 1."""

    def _make_cartesian_2d(self, n=30):
        ax = ChebyshevAxis(n, -1, 1)
        return CurvilinearGrid(
            ax, ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
            ),
        )

    def _make_cartesian_3d(self, n=15):
        ax = ChebyshevAxis(n, -1, 1)
        return CurvilinearGrid(
            ax, ax, ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
            ),
        )

    def test_laplacian_x_squared_2d(self):
        """Laplacian(x^2 + y^2) = 4 in 2D Cartesian."""
        grid = self._make_cartesian_2d()
        X, Y = grid.meshed_coords
        f = X ** 2 + Y ** 2
        result = grid.laplacian(f)
        interior = (slice(3, -3), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 4.0, decimal=1)

    def test_laplacian_xyz_squared_3d(self):
        """Laplacian(x^2 + y^2 + z^2) = 6 in 3D Cartesian."""
        grid = self._make_cartesian_3d()
        X, Y, Z = grid.meshed_coords
        f = X ** 2 + Y ** 2 + Z ** 2
        result = grid.laplacian(f)
        interior = (slice(3, -3), slice(3, -3), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 6.0, decimal=0)

    def test_gradient_x_squared_2d(self):
        """grad(x^2) = (2x, 0)."""
        grid = self._make_cartesian_2d()
        X, Y = grid.meshed_coords
        f = X ** 2
        gx, gy = grid.gradient(f)
        interior = (slice(3, -3), slice(3, -3))
        npt.assert_array_almost_equal(gx[interior], (2 * X)[interior], decimal=1)
        npt.assert_array_almost_equal(gy[interior], 0.0, decimal=1)

    def test_divergence_identity_field_2d(self):
        """div(x, y) = 2 in 2D Cartesian."""
        grid = self._make_cartesian_2d()
        X, Y = grid.meshed_coords
        result = grid.divergence(X, Y)
        interior = (slice(3, -3), slice(3, -3))
        npt.assert_array_almost_equal(result[interior], 2.0, decimal=1)

    def test_curl_3d_gradient_is_zero(self):
        """curl(grad(f)) = 0 for any scalar field."""
        grid = self._make_cartesian_3d()
        X, Y, Z = grid.meshed_coords
        f = X ** 2 * Y + Z ** 3
        gx, gy, gz = grid.gradient(f)
        cx, cy, cz = grid.curl(gx, gy, gz)
        interior = (slice(4, -4), slice(4, -4), slice(4, -4))
        for comp in (cx, cy, cz):
            npt.assert_array_almost_equal(comp[interior], 0.0, decimal=0)

    def test_curl_2d_returns_scalar(self):
        """2D curl should return a single NDArray, not a tuple."""
        grid = self._make_cartesian_2d()
        X, Y = grid.meshed_coords
        result = grid.curl(X, Y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, grid.shape)


class TestCurvilinearGridPolar(unittest.TestCase):
    """CurvilinearGrid with polar scale factors — compare to PolarGrid."""

    def _make_grids(self, nr=30, nphi=40, r_low=0.3, r_high=3):
        r_ax = create_axis(AxisType.CHEBYSHEV, nr, r_low, r_high)
        phi_ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, nphi, 0, 2 * np.pi)

        polar = PolarGrid(r_ax, phi_ax)
        curv = CurvilinearGrid(
            r_ax, phi_ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: c[0],
            ),
        )
        return polar, curv

    def test_laplacian_matches_polar(self):
        polar, curv = self._make_grids()
        R, Phi = polar.meshed_coords
        f = R ** 2 * np.cos(Phi)
        interior = (slice(3, -3), slice(None))
        npt.assert_array_almost_equal(
            curv.laplacian(f)[interior],
            polar.laplacian(f)[interior],
            decimal=1,
        )

    def test_gradient_matches_polar(self):
        polar, curv = self._make_grids()
        R, Phi = polar.meshed_coords
        f = R ** 2
        interior = (slice(3, -3), slice(None))
        g_polar = polar.gradient(f)
        g_curv = curv.gradient(f)
        for gp, gc in zip(g_polar, g_curv):
            npt.assert_array_almost_equal(gc[interior], gp[interior], decimal=1)

    def test_divergence_matches_polar(self):
        polar, curv = self._make_grids()
        R, Phi = polar.meshed_coords
        interior = (slice(3, -3), slice(None))
        npt.assert_array_almost_equal(
            curv.divergence(R, np.zeros_like(R))[interior],
            polar.divergence(R, np.zeros_like(R))[interior],
            decimal=1,
        )

    def test_curl_matches_polar(self):
        polar, curv = self._make_grids()
        R, Phi = polar.meshed_coords
        v_r = np.zeros_like(R)
        v_phi = R
        interior = (slice(3, -3), slice(None))
        npt.assert_array_almost_equal(
            curv.curl(v_r, v_phi)[interior],
            polar.curl(v_r, v_phi)[interior],
            decimal=1,
        )


class TestCurvilinearGridCylindrical(unittest.TestCase):
    """CurvilinearGrid with cylindrical scale factors — compare to CylindricalGrid."""

    def _make_grids(self, nr=20, nphi=25, nz=15):
        r_ax = create_axis(AxisType.CHEBYSHEV, nr, 0.2, 2)
        phi_ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, nphi, 0, 2 * np.pi)
        z_ax = create_axis(AxisType.CHEBYSHEV, nz, -1, 1)

        cyl = CylindricalGrid(r_ax, phi_ax, z_ax)
        curv = CurvilinearGrid(
            r_ax, phi_ax, z_ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: c[0],
                lambda c: np.ones_like(c[0]),
            ),
        )
        return cyl, curv

    def test_laplacian_matches_cylindrical(self):
        cyl, curv = self._make_grids()
        R, Phi, Z = cyl.meshed_coords
        f = R ** 2 + Z ** 2
        interior = (slice(3, -3), slice(None), slice(3, -3))
        npt.assert_array_almost_equal(
            curv.laplacian(f)[interior],
            cyl.laplacian(f)[interior],
            decimal=0,
        )

    def test_gradient_matches_cylindrical(self):
        cyl, curv = self._make_grids()
        R, Phi, Z = cyl.meshed_coords
        f = R ** 2
        interior = (slice(3, -3), slice(None), slice(3, -3))
        g_cyl = cyl.gradient(f)
        g_curv = curv.gradient(f)
        for gc, gcv in zip(g_cyl, g_curv):
            npt.assert_array_almost_equal(gcv[interior], gc[interior], decimal=1)

    def test_divergence_matches_cylindrical(self):
        cyl, curv = self._make_grids()
        R, Phi, Z = cyl.meshed_coords
        interior = (slice(3, -3), slice(None), slice(3, -3))
        npt.assert_array_almost_equal(
            curv.divergence(R, np.zeros_like(R), np.zeros_like(R))[interior],
            cyl.divergence(R, np.zeros_like(R), np.zeros_like(R))[interior],
            decimal=1,
        )


class TestCurvilinearGridSpherical(unittest.TestCase):
    """CurvilinearGrid with spherical scale factors — compare to SphericalGrid."""

    def _make_grids(self, nr=20, ntheta=15, nphi=20):
        r_ax = create_axis(AxisType.CHEBYSHEV, nr, 0.5, 2)
        theta_ax = create_axis(AxisType.CHEBYSHEV, ntheta, 0.3, np.pi - 0.3)
        phi_ax = create_axis(AxisType.EQUIDISTANT_PERIODIC, nphi, 0, 2 * np.pi)

        sph = SphericalGrid(r_ax, theta_ax, phi_ax)
        curv = CurvilinearGrid(
            r_ax, theta_ax, phi_ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: c[0],
                lambda c: c[0] * np.sin(c[1]),
            ),
        )
        return sph, curv

    def test_laplacian_matches_spherical(self):
        sph, curv = self._make_grids()
        R, Theta, Phi = sph.meshed_coords
        f = R ** 2
        interior = (slice(3, -3), slice(3, -3), slice(None))
        npt.assert_array_almost_equal(
            curv.laplacian(f)[interior],
            sph.laplacian(f)[interior],
            decimal=0,
        )

    def test_gradient_matches_spherical(self):
        sph, curv = self._make_grids()
        R, Theta, Phi = sph.meshed_coords
        f = R ** 2
        interior = (slice(3, -3), slice(3, -3), slice(None))
        g_sph = sph.gradient(f)
        g_curv = curv.gradient(f)
        for gs, gc in zip(g_sph, g_curv):
            npt.assert_array_almost_equal(gc[interior], gs[interior], decimal=1)

    def test_divergence_matches_spherical(self):
        sph, curv = self._make_grids()
        R, Theta, Phi = sph.meshed_coords
        v_r = R
        v_theta = np.zeros_like(R)
        v_phi = np.zeros_like(R)
        interior = (slice(3, -3), slice(3, -3), slice(None))
        npt.assert_array_almost_equal(
            curv.divergence(v_r, v_theta, v_phi)[interior],
            sph.divergence(v_r, v_theta, v_phi)[interior],
            decimal=1,
        )


class TestCurvilinearSingularities(unittest.TestCase):
    """Singularity handling: no NaN or Inf in output."""

    def test_no_nan_polar_near_origin(self):
        grid = PolarGrid(
            create_axis(AxisType.CHEBYSHEV, 25, 1e-6, 1),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 30, 0, 2 * np.pi),
        )
        R, Phi = grid.meshed_coords
        f = R ** 2

        for result in [grid.laplacian(f)] + list(grid.gradient(f)):
            self.assertFalse(np.any(np.isnan(result)))
            self.assertFalse(np.any(np.isinf(result)))

    def test_no_nan_spherical_near_origin(self):
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 20, 1e-6, 1),
            create_axis(AxisType.CHEBYSHEV, 15, 1e-6, np.pi - 1e-6),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 15, 0, 2 * np.pi),
        )
        R, Theta, Phi = grid.meshed_coords
        f = R ** 2
        result = grid.laplacian(f)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))


class TestCurvilinearGridCustomCoords(unittest.TestCase):
    """Test with a fully custom coordinate system (bipolar-like)."""

    def test_div_grad_equals_laplacian(self):
        r"""For any coordinate system, div(grad(f)) should equal laplacian(f).

        Uses a custom 2D system with h_1 = 1, h_2 = exp(q_1).
        """
        ax0 = ChebyshevAxis(25, 0.1, 2)
        ax1 = ChebyshevAxis(25, 0.1, 2)
        grid = CurvilinearGrid(
            ax0, ax1,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.exp(c[0]),
            ),
        )
        Q0, Q1 = grid.meshed_coords
        f = np.sin(Q0) * np.cos(Q1)

        g0, g1 = grid.gradient(f)
        div_grad = grid.divergence(g0, g1)
        lap = grid.laplacian(f)

        interior = (slice(5, -5), slice(5, -5))
        npt.assert_array_almost_equal(div_grad[interior], lap[interior], decimal=0)

    def test_custom_3d_gradient_shape(self):
        ax = ChebyshevAxis(10, 0.1, 1)
        grid = CurvilinearGrid(
            ax, ax, ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.exp(c[0]),
                lambda c: c[0] ** 2 + 1,
            ),
        )
        Q0, Q1, Q2 = grid.meshed_coords
        f = Q0 ** 2 + Q1 + Q2
        grad = grid.gradient(f)
        self.assertEqual(len(grad), 3)
        for comp in grad:
            self.assertEqual(comp.shape, grid.shape)


class TestCurvilinearCurlDimensionErrors(unittest.TestCase):
    """Curl dimension validation."""

    def test_curl_1d_raises(self):
        ax = ChebyshevAxis(10, 0, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        with self.assertRaises(ValueError):
            grid.curl(np.ones(grid.shape))

    def test_curl_4d_raises(self):
        ax = ChebyshevAxis(5, 0, 1)
        grid = CurvilinearGrid(
            ax, ax, ax, ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
            ),
        )
        with self.assertRaises(ValueError):
            grid.curl(
                np.ones(grid.shape),
                np.ones(grid.shape),
                np.ones(grid.shape),
                np.ones(grid.shape),
            )

    def test_divergence_wrong_components_raises(self):
        ax = ChebyshevAxis(10, 0, 1)
        grid = CurvilinearGrid(
            ax, ax,
            scale_factors=(
                lambda c: np.ones_like(c[0]),
                lambda c: np.ones_like(c[0]),
            ),
        )
        with self.assertRaises(ValueError):
            grid.divergence(
                np.ones(grid.shape),
                np.ones(grid.shape),
                np.ones(grid.shape),
            )


class TestSubclassesPreserveInterface(unittest.TestCase):
    """SphericalGrid, CylindricalGrid, PolarGrid still work as CurvilinearGrid."""

    def test_spherical_is_curvilinear(self):
        grid = SphericalGrid(
            create_axis(AxisType.CHEBYSHEV, 10, 0.5, 2),
            create_axis(AxisType.CHEBYSHEV, 10, 0.3, np.pi - 0.3),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0, 2 * np.pi),
        )
        self.assertIsInstance(grid, CurvilinearGrid)
        self.assertIsInstance(grid, Grid)

    def test_cylindrical_is_curvilinear(self):
        grid = CylindricalGrid(
            create_axis(AxisType.CHEBYSHEV, 10, 0.1, 2),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0, 2 * np.pi),
            create_axis(AxisType.CHEBYSHEV, 10, -1, 1),
        )
        self.assertIsInstance(grid, CurvilinearGrid)
        self.assertIsInstance(grid, Grid)

    def test_polar_is_curvilinear(self):
        grid = PolarGrid(
            create_axis(AxisType.CHEBYSHEV, 10, 0.1, 1),
            create_axis(AxisType.EQUIDISTANT_PERIODIC, 10, 0, 2 * np.pi),
        )
        self.assertIsInstance(grid, CurvilinearGrid)
        self.assertIsInstance(grid, Grid)


class TestScaleFactorCaching(unittest.TestCase):
    """Scale factors and diff ops should be lazily computed and cached."""

    def test_scale_factors_cached(self):
        ax = ChebyshevAxis(15, 0.1, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        self.assertIsNone(grid._h_arrays)
        grid.laplacian(grid.coords ** 2)
        self.assertIsNotNone(grid._h_arrays)
        cached_ref = grid._h_arrays
        grid.laplacian(grid.coords ** 2)
        self.assertIs(grid._h_arrays, cached_ref)

    def test_diff_ops_cached(self):
        ax = ChebyshevAxis(15, 0.1, 1)
        grid = CurvilinearGrid(ax, scale_factors=(lambda c: np.ones_like(c[0]),))
        self.assertIsNone(grid._diff_ops)
        grid.gradient(grid.coords ** 2)
        ops_ref = grid._diff_ops
        grid.gradient(grid.coords ** 2)
        self.assertIs(grid._diff_ops, ops_ref)


if __name__ == "__main__":
    unittest.main()
