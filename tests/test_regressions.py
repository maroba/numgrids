"""Regression tests for bugs fixed in 0.5.0."""

import numpy as np
import numpy.testing as npt
import pytest

from numgrids import (
    Diff,
    Grid,
    Interpolator,
    SphericalGrid,
    adapt,
    chebyshev,
    equidistant,
    integrate,
    logarithmic,
    periodic,
)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_chebyshev_higher_order_derivatives(order):
    grid = Grid(chebyshev(30, 0.5, 2))
    x = grid.coords
    f = x ** 7
    coeff = np.prod(np.arange(7, 7 - order, -1))
    expected = coeff * x ** (7 - order)
    npt.assert_allclose(Diff(grid, order)(f), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_chebyshev_matrix_matches_operator(order):
    grid = Grid(chebyshev(12, -1, 1), chebyshev(9, 0, 2))
    X, Y = grid.meshed_coords
    f = np.sin(X) * np.exp(Y)
    for axis in (0, 1):
        d = Diff(grid, order, axis)
        npt.assert_allclose(
            (d.as_matrix() @ f.ravel()).reshape(grid.shape), d(f), atol=1e-8
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_log_axis_higher_order_derivatives(order):
    grid = Grid(logarithmic(300, 0.1, 10))
    x = grid.coords
    f = x ** 4
    coeff = np.prod(np.arange(4, 4 - order, -1))
    expected = coeff * x ** (4 - order)
    d = Diff(grid, order, acc=6)
    npt.assert_allclose(d(f), expected, rtol=1e-6)
    npt.assert_allclose(d.as_matrix() @ f, expected, rtol=1e-6)


def test_log_axis_derivative_multidim():
    grid = Grid(equidistant(20, 0, 1), logarithmic(200, 1, 5))
    X, Y = grid.meshed_coords
    f = X * np.log(Y)
    npt.assert_allclose(Diff(grid, 2, 1)(f), -X / Y ** 2, rtol=1e-6, atol=1e-8)


def test_integrate_spherical_includes_volume_element():
    grid = SphericalGrid(chebyshev(16, 0, 2), chebyshev(16, 0, np.pi), periodic(16))
    assert integrate(grid, np.ones(grid.shape)) == pytest.approx(4 / 3 * np.pi * 8)


def test_refine_preserves_grid_type():
    grid = SphericalGrid(chebyshev(8, 0.1, 1), chebyshev(8, 0.1, 3), periodic(8))
    for refined in (grid.refine(), grid.coarsen(), grid.refine_axis(0)):
        assert type(refined) is SphericalGrid
        R = refined.meshed_coords[0]
        npt.assert_allclose(refined.laplacian(R ** 2), 6, atol=1e-6)


def test_adapt_on_curvilinear_grid():
    grid = SphericalGrid(chebyshev(6, 0.5, 1), chebyshev(6, 0.5, 2.5), periodic(6))

    def func(g):
        R, T, P = g.meshed_coords
        return g.laplacian(R ** 3)

    result = adapt(grid, func, tol=1e-4, max_iterations=5)
    assert isinstance(result.grid, SphericalGrid)


def test_periodic_interpolation_in_wraparound_gap():
    grid = Grid(periodic(20, 0, 2 * np.pi))
    f = np.sin(grid.coords)
    for method in ("linear", "cubic", "spectral"):
        inter = Interpolator(grid, f, method=method)
        tol = {"linear": 1e-1, "cubic": 1e-3, "spectral": 1e-12}[method]
        assert inter(6.2) == pytest.approx(np.sin(6.2), abs=tol)
        # Points outside one period are wrapped:
        assert inter(6.2 + 2 * np.pi) == pytest.approx(np.sin(6.2), abs=tol)


def test_interpolation_1d_many_points_returns_all():
    grid = Grid(chebyshev(30, 0, 1))
    inter = Interpolator(grid, grid.coords ** 2)
    x = np.array([0.1, 0.2, 0.3])
    npt.assert_allclose(inter(x), x ** 2, atol=1e-6)


def test_fft_matrix_matches_operator():
    grid = Grid(periodic(16, 0, 2 * np.pi), equidistant(5, 0, 1))
    X, Y = grid.meshed_coords
    f = np.sin(2 * X) * Y
    for order in (1, 2, 3):
        d = Diff(grid, order, 0)
        npt.assert_allclose((d.as_matrix() @ f.ravel()).reshape(grid.shape), d(f), atol=1e-10)


def test_fft_diff_complex_input():
    grid = Grid(periodic(32))
    x = grid.coords
    npt.assert_allclose(Diff(grid, 1)(np.exp(2j * x)), 2j * np.exp(2j * x), atol=1e-10)


def test_integral_cache_key():
    grid = Grid(chebyshev(10, 0, 1))
    assert grid.cache["integral"] is None
    integrate(grid, grid.coords)
    assert grid.cache["integral"] is not None


def test_axis_needs_two_points():
    with pytest.raises(ValueError):
        chebyshev(1)


def test_deprecated_axis_alias_warns():
    import numgrids
    with pytest.warns(DeprecationWarning):
        ax = numgrids.Axis(numgrids.AxisType.CHEBYSHEV, 10, 0, 1)
    assert isinstance(ax, numgrids.ChebyshevAxis)


def test_diff_function_not_shadowed_by_submodule():
    import numgrids
    grid = Grid(chebyshev(10, 0, 1))
    Diff(grid, 1)  # triggers lazy imports inside the package
    assert callable(numgrids.diff)
    npt.assert_allclose(numgrids.diff(grid, grid.coords ** 2), 2 * grid.coords, atol=1e-10)
