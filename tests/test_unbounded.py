"""Tests for axes on unbounded domains (half line and whole real line)."""

import json
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import airy, beta, kv

from numgrids import (
    AxisType,
    Diff,
    Grid,
    Interpolator,
    MultiGrid,
    Neumann,
    PolarGrid,
    Robin,
    SphericalGrid,
    adapt,
    chebyshev,
    colatitude,
    convergence_study,
    create_axis,
    estimate_error,
    evolve,
    load_grid,
    periodic,
    save_grid,
)
from numgrids.api import infinite, semi_infinite
from numgrids.axes import InfiniteAxis, SemiInfiniteAxis
from numgrids.boundary import BoundaryFace


def sech(x):
    e = np.exp(-np.abs(x))          # no overflow at the far-out grid points
    return 2 * e / (1 + e ** 2)


# ── the half-line axis ────────────────────────────────────────────────

class TestSemiInfiniteAxis:

    def test_points(self):
        n, L, a = 7, 2.0, 1.0
        ax = semi_infinite(n, scale=L, low=a)
        x = ax.coords
        assert len(ax) == n
        assert x[0] == a
        assert np.all(np.diff(x) > 0) and np.all(np.isfinite(x))
        # Images of the Chebyshev-Gauss-Radau points (with xi = -1, without xi = +1):
        xi = -np.cos(2 * np.pi * np.arange(n) / (2 * n - 1))
        npt.assert_allclose(ax.coords_internal, xi, atol=1e-15)
        assert ax.coords_internal[0] == -1
        npt.assert_allclose(x, a + L * (1 + xi) / (1 - xi), rtol=1e-13)
        # Half of the points lie within one scale length of the lower end:
        assert np.sum(x <= a + L) == (n + 1) // 2

    def test_domain(self):
        ax = semi_infinite(8, scale=3.0, low=-2.0)
        assert ax.low == -2.0 and ax.high == np.inf
        assert ax.scale == 3.0
        assert not ax.periodic

    def test_boundary(self):
        ax = semi_infinite(8)
        assert ax.boundary_sides == ("low",)
        assert ax.boundary == slice(1, None)

    def test_coordinate_maps(self):
        ax = semi_infinite(9, scale=2.5, low=1.0)
        npt.assert_allclose(ax.to_internal(ax.coords), ax.coords_internal, atol=1e-15)
        npt.assert_allclose(ax.from_internal(ax.coords_internal), ax.coords, rtol=1e-12)
        assert ax.to_internal(1.0) == -1 and ax.to_internal(3.5) == 0 and ax.to_internal(np.inf) == 1
        assert ax.from_internal(1.0) == np.inf
        with pytest.raises(ValueError, match="outside the domain"):
            ax.to_internal(0.5)

    def test_factories(self):
        ax = create_axis(AxisType.SEMI_INFINITE, 5, 1.0, np.inf, scale=2.0)
        assert isinstance(ax, SemiInfiniteAxis)
        assert ax == semi_infinite(5, scale=2.0, low=1.0)
        assert create_axis("semi_infinite", 5, 0, np.inf) == semi_infinite(5)
        assert semi_infinite(5, name="r").name == "r"
        assert SemiInfiniteAxis(5) == semi_infinite(5, scale=1.0, low=0.0)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="scale"):
            semi_infinite(5, scale=0.0)
        with pytest.raises(ValueError, match="scale"):
            semi_infinite(5, scale=-1.0)
        with pytest.raises(ValueError, match="scale"):
            semi_infinite(5, scale=np.inf)
        with pytest.raises(ValueError, match="infinity"):
            SemiInfiniteAxis(5, 0.0, 10.0)
        with pytest.raises(ValueError, match="finite"):
            semi_infinite(5, low=-np.inf)
        with pytest.raises(ValueError, match="at least 2"):
            semi_infinite(1)
        with pytest.raises(TypeError, match="integer"):
            semi_infinite(5.5)
        with pytest.raises(TypeError, match="Unexpected"):
            SemiInfiniteAxis(5, spacing=0.1)

    def test_repr_and_str(self):
        ax = semi_infinite(5, scale=2.0, low=1.0, name="r")
        assert repr(ax) == "SemiInfiniteAxis(num_points=5, low=1.0, high=inf, scale=2.0, name='r')"
        assert str(ax) == "SemiInfiniteAxis(5 points on [1.0, inf), scale=2.0)"

    def test_equality_and_hash(self):
        ax = semi_infinite(5, scale=2.0, low=1.0)
        assert ax == semi_infinite(5, scale=2.0, low=1.0)
        assert hash(ax) == hash(semi_infinite(5, scale=2.0, low=1.0))
        assert ax != semi_infinite(5, scale=3.0, low=1.0)
        assert ax != semi_infinite(5, scale=2.0, low=0.0)
        assert ax != semi_infinite(6, scale=2.0, low=1.0)
        assert ax != chebyshev(5, 1.0, 2.0)
        assert len({ax, semi_infinite(5, scale=2.0, low=1.0), semi_infinite(5, scale=3.0, low=1.0)}) == 2

    def test_resized_and_renamed(self):
        ax = semi_infinite(5, scale=2.0, low=1.0, name="r")
        assert ax.resized(9) == semi_infinite(9, scale=2.0, low=1.0, name="r")
        assert ax.renamed("s") == semi_infinite(5, scale=2.0, low=1.0, name="s")
        assert ax.renamed(None).name is None

    def test_quadrature_is_exact_for_mapped_polynomials(self):
        # f = (1 - xi)^2 / 4 * xi^k  =>  f dx = L / 2 * xi^k dxi
        n, L = 9, 1.5
        ax = semi_infinite(n, scale=L, low=2.0)
        xi = ax.coords_internal
        w = ax.quadrature_weights
        assert np.all(w > 0)
        for k in range(n):
            exact = L / (k + 1) if k % 2 == 0 else 0.0
            assert w @ ((1 - xi) ** 2 / 4 * xi ** k) == pytest.approx(exact, abs=1e-13)


# ── the whole-line axis ───────────────────────────────────────────────

class TestInfiniteAxis:

    def test_points(self):
        n, L, c = 8, 2.0, 0.5
        ax = infinite(n, scale=L, center=c)
        x = ax.coords
        assert len(ax) == n
        assert np.all(np.diff(x) > 0) and np.all(np.isfinite(x))
        # Images of the Chebyshev-Gauss points (no endpoints):
        xi = -np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
        npt.assert_allclose(ax.coords_internal, xi, atol=1e-15)
        npt.assert_allclose(x, c + L * xi / np.sqrt(1 - xi ** 2), rtol=1e-13)
        assert np.sum(np.abs(x - c) <= L) == n // 2

    @pytest.mark.parametrize("n", [8, 9])
    def test_points_are_symmetric(self, n):
        x = infinite(n, scale=3.0).coords
        npt.assert_array_equal(x, -x[::-1])
        if n % 2:
            assert x[n // 2] == 0

    def test_domain(self):
        ax = infinite(8, scale=3.0, center=1.0)
        assert ax.low == -np.inf and ax.high == np.inf
        assert ax.scale == 3.0 and ax.center == 1.0
        assert not ax.periodic

    def test_no_boundary(self):
        ax = infinite(8)
        assert ax.boundary_sides == ()
        assert ax.boundary == slice(None, None)

    def test_coordinate_maps(self):
        ax = infinite(9, scale=2.5, center=1.0)
        npt.assert_allclose(ax.to_internal(ax.coords), ax.coords_internal, atol=1e-15)
        npt.assert_allclose(ax.from_internal(ax.coords_internal), ax.coords, rtol=1e-12)
        npt.assert_array_equal(ax.to_internal([-np.inf, 1.0, np.inf]), [-1, 0, 1])
        with pytest.raises(ValueError, match="outside the domain"):
            ax.to_internal(np.nan)

    def test_factories(self):
        ax = create_axis(AxisType.INFINITE, 5, -np.inf, np.inf, scale=2.0, center=1.0)
        assert isinstance(ax, InfiniteAxis)
        assert ax == infinite(5, scale=2.0, center=1.0)
        assert create_axis("infinite", 5, -np.inf, np.inf) == infinite(5)
        assert infinite(5, name="x").name == "x"
        assert InfiniteAxis(5) == infinite(5, scale=1.0, center=0.0)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="scale"):
            infinite(5, scale=0.0)
        with pytest.raises(ValueError, match="center"):
            infinite(5, center=np.inf)
        with pytest.raises(ValueError, match="whole real line"):
            InfiniteAxis(5, 0.0, np.inf)
        with pytest.raises(ValueError, match="whole real line"):
            InfiniteAxis(5, -1.0, 1.0)
        with pytest.raises(ValueError, match="at least 2"):
            infinite(1)

    def test_repr_and_str(self):
        ax = infinite(5, scale=2.0, center=1.0, name="x")
        assert repr(ax) == "InfiniteAxis(num_points=5, low=-inf, high=inf, scale=2.0, center=1.0, name='x')"
        assert str(ax) == "InfiniteAxis(5 points on (-inf, inf), scale=2.0, center=1.0)"

    def test_equality_and_hash(self):
        ax = infinite(5, scale=2.0, center=1.0)
        assert ax == infinite(5, scale=2.0, center=1.0)
        assert hash(ax) == hash(infinite(5, scale=2.0, center=1.0))
        assert ax != infinite(5, scale=2.0, center=0.0)
        assert ax != infinite(5, scale=1.0, center=1.0)
        assert ax != semi_infinite(5, scale=2.0)

    def test_resized_and_renamed(self):
        ax = infinite(5, scale=2.0, center=1.0, name="x")
        assert ax.resized(9) == infinite(9, scale=2.0, center=1.0, name="x")
        assert ax.renamed("y") == infinite(5, scale=2.0, center=1.0, name="y")

    def test_quadrature_is_exact_for_mapped_polynomials(self):
        # f = (1 - xi^2) * xi^k  =>  f dx = L * xi^k / sqrt(1 - xi^2) dxi  (Gauss-Chebyshev)
        n, L = 6, 1.5
        ax = infinite(n, scale=L, center=-1.0)
        xi = ax.coords_internal
        w = ax.quadrature_weights
        assert np.all(w > 0)
        for k in range(2 * n):
            exact = L * beta((k + 1) / 2, 0.5) if k % 2 == 0 else 0.0
            assert w @ ((1 - xi ** 2) * xi ** k) == pytest.approx(exact, abs=1e-13)


# ── differentiation ───────────────────────────────────────────────────

HALF_LINE_FUNCTIONS = {
    # name: (f, f', f'', scale)
    "exp(-x)": (lambda x: np.exp(-x), lambda x: -np.exp(-x), lambda x: np.exp(-x), 8.0),
    "x exp(-x)": (lambda x: x * np.exp(-x), lambda x: (1 - x) * np.exp(-x), lambda x: (x - 2) * np.exp(-x), 8.0),
    "1/(1+x)^2": (lambda x: (1 + x) ** -2.0, lambda x: -2 * (1 + x) ** -3.0, lambda x: 6 * (1 + x) ** -4.0, 2.0),
    "tanh(x)": (np.tanh, lambda x: sech(x) ** 2, lambda x: -2 * np.tanh(x) * sech(x) ** 2, 4.0),
}

WHOLE_LINE_FUNCTIONS = {
    "exp(-x^2)": (lambda x: np.exp(-x ** 2), lambda x: -2 * x * np.exp(-x ** 2),
                  lambda x: (4 * x ** 2 - 2) * np.exp(-x ** 2), 4.0),
    "sech(x)": (sech, lambda x: -sech(x) * np.tanh(x), lambda x: sech(x) * (1 - 2 * sech(x) ** 2), 4.0),
    "1/(1+x^2)": (lambda x: 1 / (1 + x ** 2), lambda x: -2 * x / (1 + x ** 2) ** 2,
                  lambda x: (6 * x ** 2 - 2) / (1 + x ** 2) ** 3, 2.0),
    "tanh(x)": (np.tanh, lambda x: sech(x) ** 2, lambda x: -2 * np.tanh(x) * sech(x) ** 2, 4.0),
}


class TestDerivatives:

    @pytest.mark.parametrize("name", HALF_LINE_FUNCTIONS)
    def test_half_line(self, name):
        f, df, d2f, scale = HALF_LINE_FUNCTIONS[name]
        grid = Grid(semi_infinite(64, scale=scale))
        x = grid.coords
        npt.assert_allclose(grid.diff(f(x)), df(x), atol=1e-11)
        npt.assert_allclose(grid.diff(f(x), 2), d2f(x), atol=1e-7)

    @pytest.mark.parametrize("name", WHOLE_LINE_FUNCTIONS)
    def test_whole_line(self, name):
        f, df, d2f, scale = WHOLE_LINE_FUNCTIONS[name]
        grid = Grid(infinite(80, scale=scale))
        x = grid.coords
        npt.assert_allclose(grid.diff(f(x)), df(x), atol=1e-10)
        npt.assert_allclose(grid.diff(f(x), 2), d2f(x), atol=1e-9)

    def test_rational_functions_are_exact(self):
        # 1/(1+x)^2 and 1/(1+x^2) are low-degree polynomials in xi for scale = 1.
        grid = Grid(semi_infinite(8, scale=1.0))
        x = grid.coords
        npt.assert_allclose(grid.diff((1 + x) ** -2.0), -2 * (1 + x) ** -3.0, atol=1e-13)
        grid = Grid(infinite(8, scale=1.0))
        x = grid.coords
        npt.assert_allclose(grid.diff(1 / (1 + x ** 2)), -2 * x / (1 + x ** 2) ** 2, atol=1e-13)

    def test_shifted_domains(self):
        grid = Grid(semi_infinite(64, scale=8.0, low=-3.0))
        x = grid.coords
        npt.assert_allclose(grid.diff(np.exp(-(x + 3))), -np.exp(-(x + 3)), atol=1e-11)
        grid = Grid(infinite(80, scale=4.0, center=2.0))
        x = grid.coords
        npt.assert_allclose(grid.diff(np.exp(-(x - 2) ** 2)), -2 * (x - 2) * np.exp(-(x - 2) ** 2), atol=1e-10)

    @pytest.mark.parametrize("make, f, df", [
        (lambda n: semi_infinite(n, scale=4.0), lambda x: np.exp(-x), lambda x: -np.exp(-x)),
        (lambda n: infinite(n, scale=3.0), sech, lambda x: -sech(x) * np.tanh(x)),
    ])
    def test_spectral_convergence(self, make, f, df):
        errors = []
        for n in (12, 24, 48):
            grid = Grid(make(n))
            x = grid.coords
            errors.append(np.abs(grid.diff(f(x)) - df(x)).max())
        # Much faster than any fixed algebraic order:
        assert errors[1] < errors[0] / 50
        assert errors[2] < errors[1] / 500

    @pytest.mark.parametrize("axis", [semi_infinite(40, scale=2.0), infinite(40, scale=2.0)])
    def test_constants_have_zero_derivative(self, axis):
        grid = Grid(axis)
        npt.assert_allclose(grid.diff(np.ones(grid.shape)), 0, atol=1e-12)
        npt.assert_allclose(grid.diff(np.ones(grid.shape), 2), 0, atol=1e-8)

    def test_higher_derivatives(self):
        grid = Grid(semi_infinite(48, scale=8.0))
        x = grid.coords
        npt.assert_allclose(grid.diff(np.exp(-x), 3), -np.exp(-x), atol=1e-6)
        grid = Grid(infinite(80, scale=4.0))
        x = grid.coords
        f = np.exp(-x ** 2)
        npt.assert_allclose(grid.diff(f, 3), (12 * x - 8 * x ** 3) * f, atol=1e-9)
        npt.assert_allclose(grid.diff(f, 4), (16 * x ** 4 - 48 * x ** 2 + 12) * f, atol=1e-8)

    def test_second_derivative_of_algebraically_decaying_function(self):
        # f' is not a polynomial in xi (it has a factor (1 - xi^2)^(3/2)), so applying the
        # first derivative twice converges slowly; the second derivative itself does not.
        grid = Grid(infinite(80, scale=2.0))
        x = grid.coords
        f = 1 / (1 + x ** 2)
        exact = (6 * x ** 2 - 2) / (1 + x ** 2) ** 3
        assert np.abs(grid.diff(f, 2) - exact).max() < 1e-11
        assert np.abs(grid.diff(grid.diff(f)) - exact).max() > 1e-9

    def test_matrix_matches_operator(self):
        grid = Grid(semi_infinite(12, scale=2.0), infinite(10, scale=2.0))
        R, X = grid.meshed_coords
        f = np.exp(-R - X ** 2)
        for order in (1, 2):
            for axis in (0, 1):
                d = Diff(grid, order, axis)
                assert d.as_matrix().shape == (grid.size, grid.size)
                npt.assert_allclose((d.as_matrix() @ f.ravel()).reshape(grid.shape), d(f), atol=1e-10)

    def test_partial_derivatives_in_mixed_grid(self):
        grid = Grid(x=infinite(60, scale=4.0), r=semi_infinite(48, scale=8.0), y=chebyshev(12, 0, 1))
        X, R, Y = grid.meshed_coords
        f = np.exp(-X ** 2 - R) * Y ** 2
        npt.assert_allclose(grid.diff(f, axis="x"), -2 * X * f, atol=1e-8)
        npt.assert_allclose(grid.diff(f, axis="r"), -f, atol=1e-8)
        npt.assert_allclose(grid.diff(f, axis="y"), 2 * np.exp(-X ** 2 - R) * Y, atol=1e-8)
        lap = (4 * X ** 2 - 2 + 1) * f + 2 * np.exp(-X ** 2 - R)
        npt.assert_allclose(grid.laplacian(f), lap, atol=1e-6)

    def test_complex_fields(self):
        grid = Grid(infinite(80, scale=4.0))
        x = grid.coords
        f = np.exp(-x ** 2 + 1j * x)
        npt.assert_allclose(grid.diff(f), (-2 * x + 1j) * f, atol=1e-9)


# ── integration ───────────────────────────────────────────────────────

class TestIntegration:

    def test_half_line(self):
        grid = Grid(semi_infinite(48, scale=8.0))
        x = grid.coords
        assert grid.integrate(np.exp(-x)) == pytest.approx(1.0, abs=1e-13)
        assert grid.integrate(x ** 2 * np.exp(-x)) == pytest.approx(2.0, abs=1e-12)
        assert grid.integrate(np.exp(-x ** 2)) == pytest.approx(np.sqrt(np.pi) / 2, abs=1e-9)
        assert grid.integrate(1 / (1 + x) ** 2) == pytest.approx(1.0, abs=1e-12)

    def test_half_line_shifted(self):
        grid = Grid(semi_infinite(48, scale=8.0, low=2.0))
        x = grid.coords
        assert grid.integrate(np.exp(-x)) == pytest.approx(np.exp(-2.0), abs=1e-13)

    def test_whole_line(self):
        grid = Grid(infinite(64, scale=4.0))
        x = grid.coords
        assert grid.integrate(np.exp(-x ** 2)) == pytest.approx(np.sqrt(np.pi), abs=1e-13)
        assert grid.integrate(x ** 2 * np.exp(-x ** 2)) == pytest.approx(np.sqrt(np.pi) / 2, abs=1e-13)
        assert grid.integrate(sech(x)) == pytest.approx(np.pi, abs=1e-11)
        assert grid.integrate(np.exp(-x ** 2 + 1j * x)) == pytest.approx(np.sqrt(np.pi) * np.exp(-0.25), abs=1e-12)

    @pytest.mark.parametrize("n", [4, 16, 64])
    def test_lorentzian(self, n):
        # 1/(1 + x^2) = 1 - xi^2 for scale = 1: exact for any number of points
        grid = Grid(infinite(n, scale=1.0))
        assert grid.integrate(1 / (1 + grid.coords ** 2)) == pytest.approx(np.pi, abs=1e-13)

    def test_lorentzian_with_other_scale(self):
        grid = Grid(infinite(64, scale=3.0))
        assert grid.integrate(1 / (1 + grid.coords ** 2)) == pytest.approx(np.pi, abs=1e-10)

    def test_convergence(self):
        errors = []
        for n in (8, 16, 32):
            grid = Grid(semi_infinite(n, scale=8.0))
            errors.append(abs(grid.integrate(grid.coords * np.exp(-grid.coords)) - 1))
        assert errors[1] < errors[0] / 50
        assert errors[2] < errors[1] / 500

    def test_mixed_grid(self):
        grid = Grid(x=infinite(48, scale=4.0), r=semi_infinite(40, scale=8.0), y=chebyshev(10, 0, 2))
        X, R, Y = grid.meshed_coords
        assert grid.integrate(np.exp(-X ** 2 - R) * Y) == pytest.approx(2 * np.sqrt(np.pi), abs=1e-11)
        # Integral over the face r = 0:
        face = grid.faces["r_low"]
        assert face.integrate(np.exp(-X ** 2 - R) * Y) == pytest.approx(2 * np.sqrt(np.pi), abs=1e-11)


# ── eigenvalue problems ───────────────────────────────────────────────

class TestEigenproblems:

    def test_harmonic_oscillator(self):
        grid = Grid(x=infinite(64, scale=5.0))
        x = grid.coords
        H = -0.5 * grid.diff_operator(2) + 0.5 * x ** 2
        E, psi = H.eig(k=8, which="SR")
        npt.assert_allclose(E, np.arange(8) + 0.5, atol=1e-10)
        # Ground state (normalized with the grid's quadrature):
        psi0 = psi[0] / np.sqrt(grid.integrate(psi[0] ** 2))
        npt.assert_allclose(psi0, np.pi ** -0.25 * np.exp(-x ** 2 / 2), atol=1e-9)
        # Eigenfunctions are orthogonal:
        assert grid.integrate(psi[1] * psi[3]) == pytest.approx(0, abs=1e-10)

    def test_harmonic_oscillator_spectrum_is_real(self):
        # No boundary condition is needed and no spurious modes show up below the true levels.
        grid = Grid(infinite(40, scale=4.0))
        H = -0.5 * grid.diff_operator(2) + 0.5 * grid.coords ** 2
        ev = np.linalg.eigvals(H.as_matrix().toarray())
        assert np.abs(ev.imag).max() < 1e-8
        assert ev.real.min() == pytest.approx(0.5, abs=1e-8)

    def test_hydrogen_levels(self):
        grid = Grid(r=semi_infinite(48, scale=10.0))
        r = grid.coords
        V = np.zeros(grid.shape)
        V[1:] = -1 / r[1:]            # the value at r = 0 is irrelevant: u(0) = 0
        H = -0.5 * grid.diff_operator(2) + V
        E, u = H.eig(k=3, bcs=0, which="SR")
        npt.assert_allclose(E, -0.5 / np.arange(1, 4) ** 2, atol=1e-10)
        u1 = u[0] / np.sqrt(grid.integrate(u[0] ** 2))
        npt.assert_allclose(u1, 2 * r * np.exp(-r), atol=1e-9)

    def test_sech_squared_well_with_neumann_condition(self):
        # -u'' - 2 sech^2(x) u = E u on [0, inf) with u'(0) = 0: the even bound state E = -1, u = sech(x)
        grid = Grid(x=semi_infinite(48, scale=4.0))
        x = grid.coords
        H = -grid.diff_operator(2) - 2 * sech(x) ** 2
        E, u = H.eig(k=1, bcs=Neumann(0), which="SR")
        assert E[0] == pytest.approx(-1.0, abs=1e-9)
        npt.assert_allclose(u[0] / u[0][0], sech(x), atol=1e-8)

    def test_waveguide_2d(self):
        # -lap u + x^2 u = lambda u on the strip R x [0, 1], u = 0 on the walls:
        # lambda = (2 n + 1) + (m pi)^2
        grid = Grid(x=infinite(36, scale=3.0), y=chebyshev(12, 0, 1))
        X, _ = grid.meshed_coords
        assert list(grid.faces) == ["1_low", "1_high"]
        H = -grid.laplacian_operator() + X ** 2
        vals, _ = H.eig(k=5, bcs=0, which="SR")
        npt.assert_allclose(vals, 2 * np.arange(5) + 1 + np.pi ** 2, atol=1e-5)

    def test_oscillator_2d(self):
        grid = Grid(x=infinite(18, scale=3.0), y=infinite(18, scale=3.0))
        X, Y = grid.meshed_coords
        H = -0.5 * grid.laplacian_operator() + 0.5 * (X ** 2 + Y ** 2)
        vals, vecs = H.eig(k=3, which="SR")
        npt.assert_allclose(vals, [1, 2, 2], atol=1e-3)
        assert np.isrealobj(vecs)


# ── boundary value problems ───────────────────────────────────────────

class TestBoundaryValueProblems:

    @pytest.fixture
    def grid(self):
        return Grid(x=semi_infinite(40, scale=4.0))

    def test_dirichlet(self, grid):
        # u'' - u = 0, u(0) = 1; the growing solution exp(x) is excluded automatically
        u = (grid.diff_operator(2) - 1).solve(0, bcs={"x_low": 1.0})
        npt.assert_allclose(u, np.exp(-grid.coords), atol=1e-10)

    def test_scalar_bcs_apply_to_the_only_face(self, grid):
        u = (grid.diff_operator(2) - 1).solve(0, bcs=1.0)
        npt.assert_allclose(u, np.exp(-grid.coords), atol=1e-10)

    def test_neumann(self, grid):
        # outward normal derivative at x = 0 is -u'(0) = 1
        u = (grid.diff_operator(2) - 1).solve(0, bcs={"x_low": Neumann(1.0)})
        npt.assert_allclose(u, np.exp(-grid.coords), atol=1e-10)

    def test_robin(self, grid):
        # u - u' = 3 at x = 0, solved by 1.5 exp(-x)
        u = (grid.diff_operator(2) - 1).solve(0, bcs={"x_low": Robin(1.0, 1.0, 3.0)})
        npt.assert_allclose(u, 1.5 * np.exp(-grid.coords), atol=1e-10)

    def test_inhomogeneous(self, grid):
        # u'' - u = -2 exp(-x), u(0) = 0  =>  u = x exp(-x)
        x = grid.coords
        u = (grid.diff_operator(2) - 1).solve(-2 * np.exp(-x), bcs=0)
        npt.assert_allclose(u, x * np.exp(-x), atol=1e-9)

    def test_airy_equation(self, grid):
        # u'' - x u = 0 with the value of the decaying Airy function at x = 0: that
        # solution is found, not the growing one
        x = grid.coords
        u = (grid.diff_operator(2) - x).solve(0, bcs=airy(0)[0])
        npt.assert_allclose(u, airy(x)[0], atol=1e-10)

    def test_poisson_equation_selects_the_bounded_solution(self, grid):
        # u'' = exp(-x), u(0) = 1: of the solutions exp(-x) + c x only c = 0 is bounded
        x = grid.coords
        u = grid.diff_operator(2).solve(np.exp(-x), bcs=1.0)
        npt.assert_allclose(u, np.exp(-x), atol=1e-9)

    def test_whole_line(self):
        # u'' - u = f without any boundary condition
        grid = Grid(x=infinite(64, scale=4.0))
        x = grid.coords
        u = (grid.diff_operator(2) - 1).solve((4 * x ** 2 - 3) * np.exp(-x ** 2))
        npt.assert_allclose(u, np.exp(-x ** 2), atol=1e-10)

    def test_half_plane_with_periodic_axis(self):
        # lap u - u = 0 for x > 0, u(0, y) = cos(y)  =>  u = exp(-sqrt(2) x) cos(y)
        grid = Grid(x=semi_infinite(32, scale=4.0), y=periodic(16))
        X, Y = grid.meshed_coords
        u = (grid.laplacian_operator() - 1).solve(0, bcs={"x_low": lambda c: np.cos(c[1])})
        npt.assert_allclose(u, np.exp(-np.sqrt(2) * X) * np.cos(Y), atol=1e-9)

    def test_missing_faces(self, grid):
        with pytest.raises(KeyError, match="x_high"):
            grid.diff_operator(2).solve(0, bcs={"x_high": 0})
        with pytest.raises(ValueError, match="no 'high' boundary face"):
            BoundaryFace(grid, 0, "high")
        with pytest.raises(ValueError, match="no 'low' boundary face"):
            BoundaryFace(Grid(infinite(8)), 0, "low")


# ── time evolution ────────────────────────────────────────────────────

class TestEvolution:

    def test_heat_equation_on_the_whole_line(self):
        # A Gaussian stays a Gaussian and spreads.
        grid = Grid(x=infinite(48, scale=4.0))
        x = grid.coords
        D, t = 0.5, 1.0
        res = evolve(grid, D * grid.laplacian_operator(), np.exp(-x ** 2), (0, t),
                     method="BDF", t_eval=[0, t], rtol=1e-8, atol=1e-10)
        assert res.success
        exact = np.exp(-x ** 2 / (1 + 4 * D * t)) / np.sqrt(1 + 4 * D * t)
        npt.assert_allclose(res.final, exact, atol=1e-6)
        # Heat is conserved:
        assert grid.integrate(res.final) == pytest.approx(np.sqrt(np.pi), abs=1e-6)

    def test_heat_equation_on_the_half_line(self):
        grid = Grid(x=semi_infinite(48, scale=4.0))
        x = grid.coords
        t = 0.5
        res = evolve(grid, grid.laplacian_operator(), x * np.exp(-x ** 2), (0, t), bcs=0,
                     method="BDF", t_eval=[0, t], rtol=1e-8, atol=1e-10)
        assert res.success
        s = 1 + 4 * t
        npt.assert_allclose(res.final, x * np.exp(-x ** 2 / s) / s ** 1.5, atol=1e-6)

    def test_advection_with_explicit_method(self):
        # u_t = -u_x: the pulse moves to the right; the spectrum of d/dx is purely imaginary.
        grid = Grid(x=infinite(64, scale=4.0))
        x = grid.coords
        res = evolve(grid, lambda t, u: -grid.diff(u), np.exp(-x ** 2), (0, 1), t_eval=[0, 1],
                     rtol=1e-8, atol=1e-10)
        assert res.success
        npt.assert_allclose(res.final, np.exp(-(x - 1) ** 2), atol=1e-6)

    @pytest.mark.parametrize("axis", [semi_infinite(32, scale=2.0), infinite(32, scale=2.0)])
    def test_diffusion_operator_is_stable(self, axis):
        grid = Grid(axis)
        L, _ = grid.laplacian_operator().with_bcs(0.0, bcs=0)
        interior = ~grid.boundary.ravel()
        ev = np.linalg.eigvals(L.toarray()[np.ix_(interior, interior)])
        assert np.abs(ev.imag).max() < 1e-8 * np.abs(ev).max()
        assert ev.real.max() < 1e-8


# ── interpolation ─────────────────────────────────────────────────────

class TestInterpolation:

    def test_spectral_half_line(self):
        grid = Grid(semi_infinite(40, scale=4.0, low=1.0))
        x = grid.coords
        f = np.exp(-(x - 1))
        npt.assert_array_equal(grid.interpolate(f, x, method="spectral"), f)
        # Between the points and far beyond the outermost point:
        pts = np.array([1.0, 1.3, 2.7, 11.0, 51.0, 2 * x[-1], 1e9])
        npt.assert_allclose(grid.interpolate(f, pts, method="spectral"), np.exp(-(pts - 1)), atol=1e-10)
        assert grid.interpolate(f, 2.5, method="spectral") == pytest.approx(np.exp(-1.5), abs=1e-10)

    def test_spectral_whole_line(self):
        grid = Grid(infinite(64, scale=4.0, center=1.0))
        x = grid.coords
        f = np.exp(-(x - 1) ** 2)
        npt.assert_array_equal(grid.interpolate(f, x, method="spectral"), f)
        pts = np.array([-1e6, 3 * x[0], -30.0, -1.2, 0.0, 0.3, 1.7, 10.0, 3 * x[-1], 1e6])
        npt.assert_allclose(grid.interpolate(f, pts, method="spectral"), np.exp(-(pts - 1) ** 2), atol=1e-10)

    def test_value_at_infinity(self):
        grid = Grid(semi_infinite(32, scale=2.0))
        x = grid.coords
        f = 3 + 1 / (1 + x)          # tends to 3
        assert grid.interpolate(f, np.inf, method="spectral") == pytest.approx(3.0, abs=1e-12)
        grid = Grid(infinite(32, scale=2.0))
        values = grid.interpolate(np.tanh(grid.coords), [-np.inf, np.inf], method="spectral")
        npt.assert_allclose(values, [-1, 1], atol=1e-6)

    @pytest.mark.parametrize("method, tol", [("linear", 3e-2), ("quadratic", 5e-4), ("cubic", 2e-4)])
    def test_spline_methods(self, method, tol):
        grid = Grid(semi_infinite(40, scale=4.0))
        x = grid.coords
        pts = np.array([0.0, 0.3, 1.7, 10.0, 50.0, x[-1], 3 * x[-1], np.inf])
        values = grid.interpolate(np.exp(-x), pts, method=method)
        npt.assert_allclose(values, np.exp(-pts), atol=tol)
        grid = Grid(infinite(64, scale=4.0))
        x = grid.coords
        pts = np.array([-np.inf, 3 * x[0], x[0], -1.2, 0.0, 0.3, 1.7, 10.0, 3 * x[-1]])
        values = grid.interpolate(np.exp(-x ** 2), pts, method=method)
        npt.assert_allclose(values, np.exp(-pts ** 2), atol=tol)

    def test_default_method(self):
        grid = Grid(semi_infinite(40, scale=4.0))
        x = grid.coords
        assert grid.interpolate(np.exp(-x), 0.7) == pytest.approx(np.exp(-0.7), abs=1e-5)

    @pytest.mark.parametrize("method", ["spectral", "linear", "cubic"])
    def test_outside_the_domain(self, method):
        grid = Grid(semi_infinite(16, low=1.0))
        with pytest.raises(ValueError, match="outside the domain"):
            grid.interpolate(np.exp(-grid.coords), 0.5, method=method)

    @pytest.mark.parametrize("method, tol", [("spectral", 1e-7), ("linear", 1e-2), ("cubic", 1e-4)])
    def test_mixed_grid(self, method, tol):
        grid = Grid(x=infinite(40, scale=3.0), y=chebyshev(16, 0, 1), r=semi_infinite(32, scale=4.0, low=1.0))
        X, Y, R = grid.meshed_coords
        f = np.exp(-X ** 2 - (R - 1)) * np.cos(Y)
        inter = Interpolator(grid, f, method=method)
        pts = [(0.4, 0.3, 2.5), (-1.5, 0.9, 1.0), (40.0, 0.5, 1e4)]
        exact = [np.exp(-px ** 2 - (pr - 1)) * np.cos(py) for px, py, pr in pts]
        npt.assert_allclose(inter(pts), exact, atol=tol)

    @pytest.mark.parametrize("method, tol", [("spectral", 5e-9), ("cubic", 2e-4)])
    def test_onto_another_grid(self, method, tol):
        grid = Grid(semi_infinite(40, scale=4.0), infinite(48, scale=3.0))
        func = lambda g: np.exp(-g.meshed_coords[0] - g.meshed_coords[1] ** 2)
        # The finer grid reaches further out than the coarse one.
        for target in (grid.refine(), grid.coarsen(), Grid(semi_infinite(20, scale=1.0), infinite(20, scale=5.0))):
            values = grid.interpolate(func(grid), target, method=method)
            assert values.shape == target.shape
            npt.assert_allclose(values, func(target), atol=tol)


# ── integration with the rest of the library ──────────────────────────

class TestGrids:

    @pytest.fixture
    def grid(self):
        return Grid(x=infinite(10, scale=3.0, center=1.0), y=chebyshev(6, 0, 1), r=semi_infinite(8, scale=2.0, low=1.0))

    def test_faces_and_masks(self, grid):
        assert list(grid.faces) == ["1_low", "1_high", "2_low"]
        assert "r_low" in grid.faces and "r_high" not in grid.faces and "x_low" not in grid.faces
        expected = np.zeros(grid.shape, dtype=bool)
        expected[:, 0, :] = expected[:, -1, :] = expected[:, :, 0] = True
        npt.assert_array_equal(grid.boundary, expected)
        npt.assert_array_equal(grid.interior, ~expected)
        face = grid.faces["r_low"]
        assert face.normal_sign == -1
        npt.assert_array_equal(face.coords[2], 1.0)

    def test_no_boundary_at_all(self):
        grid = Grid(infinite(8), infinite(6))
        assert list(grid.faces) == []
        assert not grid.boundary.any()

    def test_named_axes(self, grid):
        assert grid.get_axis("r") == semi_infinite(8, scale=2.0, low=1.0, name="r")
        assert grid.axis_index("x") == 0
        f = grid.evaluate(lambda x, y, r: np.exp(-(x - 1) ** 2 - r) * y)
        npt.assert_allclose(grid.diff(f, axis="y"), grid.diff(f, axis=1))

    def test_repr(self, grid):
        r = repr(grid)
        assert "InfiniteAxis(num_points=10, low=-inf, high=inf, scale=3.0, center=1.0, name='x')" in r
        assert "SemiInfiniteAxis(num_points=8, low=1.0, high=inf, scale=2.0, name='r')" in r

    def test_equality(self, grid):
        same = Grid(x=infinite(10, scale=3.0, center=1.0), y=chebyshev(6, 0, 1),
                    r=semi_infinite(8, scale=2.0, low=1.0))
        other = Grid(x=infinite(10, scale=3.0, center=1.0), y=chebyshev(6, 0, 1),
                     r=semi_infinite(8, scale=2.5, low=1.0))
        assert grid == same and hash(grid) == hash(same)
        assert grid != other

    def test_refinement(self, grid):
        fine = grid.refine()
        assert fine.shape == (20, 12, 16)
        assert fine.axes[0] == infinite(20, scale=3.0, center=1.0, name="x")
        assert fine.axes[2] == semi_infinite(16, scale=2.0, low=1.0, name="r")
        assert grid.coarsen().shape == (5, 3, 4)
        assert grid.refine_axis("r", 1.5).shape == (10, 6, 12)
        assert grid.refine_axis(0, 0.5).axes[0] == infinite(5, scale=3.0, center=1.0, name="x")

    def test_multigrid(self):
        mg = MultiGrid(semi_infinite(32, scale=2.0), infinite(32, scale=2.0))
        assert [level.shape for level in mg.levels] == [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]
        assert mg.levels[1].axes[0] == semi_infinite(16, scale=2.0)
        func = lambda g: np.exp(-g.meshed_coords[0] - g.meshed_coords[1] ** 2)
        coarse = mg.transfer(func(mg.levels[0]), 0, 1)
        npt.assert_allclose(coarse, func(mg.levels[1]), atol=0.03)
        # Prolongation needs values beyond the outermost points of the coarse grid:
        fine = mg.transfer(coarse, 1, 0)
        npt.assert_allclose(fine, func(mg.levels[0]), atol=0.1)

    def test_save_and_load(self, grid, tmp_path):
        f = grid.evaluate(lambda x, y, r: np.exp(-x ** 2 - r) * y)
        save_grid(tmp_path / "unbounded.npz", grid, f=f)
        loaded, data = load_grid(tmp_path / "unbounded.npz")
        assert loaded == grid
        assert isinstance(loaded.axes[0], InfiniteAxis) and isinstance(loaded.axes[2], SemiInfiniteAxis)
        npt.assert_array_equal(loaded.axes[2].coords, grid.axes[2].coords)
        npt.assert_array_equal(data["f"], f)

    def test_xarray_round_trip(self, grid):
        pytest.importorskip("xarray")
        f = grid.evaluate(lambda x, y, r: np.exp(-x ** 2 - r) * y)
        ds = grid.to_xarray(f=f)
        assert list(ds.dims) == ["x", "y", "r"]
        npt.assert_array_equal(ds["r"].values, grid.axes[2].coords)

        def reject(constant):
            raise AssertionError(f"not valid JSON: {constant}")

        # The metadata is strict JSON (no Infinity):
        meta = json.loads(ds.attrs["numgrids_grid"], parse_constant=reject)
        assert meta["axes"][0]["low"] is None and meta["axes"][2]["high"] is None
        loaded, data = Grid.from_xarray(ds)
        assert loaded == grid
        npt.assert_array_equal(data["f"], f)
        # A single DataArray keeps the grid definition, too:
        loaded, _ = Grid.from_xarray(ds["f"])
        assert loaded == grid

    def test_xarray_without_metadata_cannot_be_inferred(self):
        xr = pytest.importorskip("xarray")
        ax = semi_infinite(8)
        da = xr.DataArray(np.exp(-ax.coords), coords={"r": ax.coords}, dims="r", name="f")
        with pytest.raises(ValueError, match="Cannot infer the axis type"):
            Grid.from_xarray(da)

    def test_adaptive_refinement(self):
        func = lambda g: np.exp(-g.meshed_coords[0] ** 2) * g.meshed_coords[1]
        grid = Grid(infinite(8, scale=3.0), chebyshev(4, 0, 1))
        report = estimate_error(grid, func)
        assert report["per_axis"][0] > 1e-3 > report["per_axis"][1]
        result = adapt(grid, func, tol=1e-3)
        assert result.converged
        assert result.grid.shape[0] > 8 and result.grid.shape[1] == 4
        assert isinstance(result.grid.axes[0], InfiniteAxis)

    def test_convergence_study(self):
        # Without an exact solution, the finest grid is interpolated onto the coarser ones.
        res = convergence_study(
            lambda n: Grid(infinite(n, scale=4.0)),
            lambda g: g.diff(np.exp(-g.coords ** 2)),
            ns=[16, 32, 64],
        )
        assert res.errors[1] < res.errors[0] / 1000
        res = convergence_study(
            lambda n: Grid(semi_infinite(n, scale=8.0)),
            lambda g: g.integrate(g.coords * np.exp(-g.coords)),
            ns=[8, 16, 32],
            exact=1.0,
        )
        assert res.order > 8

    def test_adaptive_refinement_half_line(self):
        result = adapt(Grid(semi_infinite(8, scale=4.0)), lambda g: np.exp(-g.coords), tol=1e-3)
        assert result.converged and len(result.grid.axes[0]) > 8


class TestPlotting:

    @pytest.fixture(autouse=True)
    def _matplotlib(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        yield
        import matplotlib.pyplot as plt
        plt.close("all")

    @pytest.mark.parametrize("axis", [semi_infinite(12, scale=2.0), infinite(12, scale=2.0)])
    def test_plot_1d(self, axis):
        grid = Grid(x=axis)
        ax = grid.plot(np.exp(-grid.coords ** 2))
        line, = ax.get_lines()
        npt.assert_array_equal(line.get_xdata(), grid.coords)

    def test_plot_2d_and_3d(self):
        grid = Grid(x=infinite(10), r=semi_infinite(8))
        assert grid.plot(grid.evaluate(lambda x, r: np.exp(-x ** 2 - r))) is not None
        grid = Grid(x=infinite(6), r=semi_infinite(5), y=chebyshev(4))
        assert grid.plot(np.ones(grid.shape), fixed={"y": 1}) is not None

    def test_plot_grid_points(self):
        with patch("matplotlib.pyplot.show"):
            Grid(infinite(6), semi_infinite(5)).plot()

    @pytest.mark.parametrize("axis", [semi_infinite(12, scale=2.0), infinite(12, scale=2.0)])
    def test_axis_plot(self, axis):
        with patch("matplotlib.pyplot.show") as show:
            axis.plot()
        show.assert_called_once()


# ── curvilinear grids ─────────────────────────────────────────────────

class TestCurvilinear:

    def test_exterior_of_a_disk(self):
        # lap u - u = 0 outside the unit disk, u = cos(2 phi) on the rim:
        # u = K_2(r) / K_2(1) cos(2 phi) with the modified Bessel function K_2
        grid = PolarGrid(semi_infinite(32, scale=2.0, low=1.0), periodic(16))
        R, Phi = grid.meshed_coords
        assert list(grid.faces) == ["0_low"]
        u = (grid.laplacian_operator() - 1).solve(0, bcs={"r_low": lambda c: np.cos(2 * c[1])})
        npt.assert_allclose(u, kv(2, R) / kv(2, 1.0) * np.cos(2 * Phi), atol=1e-9)

    def test_laplacian_in_polar_coordinates(self):
        grid = PolarGrid(semi_infinite(48, scale=6.0, low=1.0), periodic(16))
        R, Phi = grid.meshed_coords
        f = np.exp(-R) * np.cos(Phi)
        exact = (1 - 1 / R - 1 / R ** 2) * f
        npt.assert_allclose(grid.laplacian(f), exact, atol=1e-8)
        npt.assert_allclose(grid.laplacian_operator()(f), exact, atol=1e-8)
        npt.assert_allclose(grid.divergence(*grid.gradient(f)), exact, atol=1e-7)

    def test_exterior_of_a_sphere(self):
        # Screened Poisson (Yukawa) equation outside the unit sphere: u = exp(-(r - 1)) / r
        grid = SphericalGrid(semi_infinite(20, scale=2.0, low=1.0), colatitude(8), periodic(8))
        R, _, _ = grid.meshed_coords
        assert list(grid.faces) == ["0_low"]
        u = (grid.laplacian_operator() - 1).solve(0, bcs=1.0)
        npt.assert_allclose(u, np.exp(-(R - 1)) / R, atol=1e-6)
        # Volume integral of exp(-(r - 1)) over r > 1 and the area of the unit sphere:
        assert grid.integrate(np.exp(-(R - 1))) == pytest.approx(20 * np.pi, rel=1e-5)
        assert grid.faces["r_low"].integrate(np.ones(grid.shape)) == pytest.approx(4 * np.pi, rel=1e-12)

    def test_spherical_grid_from_the_origin(self):
        # Like a Chebyshev axis starting at r = 0: a grid point lies on the coordinate singularity.
        grid = SphericalGrid(semi_infinite(48, scale=4.0), colatitude(8), periodic(8))
        R, _, _ = grid.meshed_coords
        f = np.exp(-R ** 2)
        assert grid.integrate(f) == pytest.approx(np.pi ** 1.5, rel=1e-8)
        lap = grid.laplacian(f)
        npt.assert_allclose(lap[1:], ((4 * R ** 2 - 6) * f)[1:], atol=1e-4)
        npt.assert_array_equal(lap[0], 0)        # singular row at r = 0
