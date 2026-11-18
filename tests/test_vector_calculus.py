"""Vector calculus across the origin and the poles, and its operator forms."""

import numpy as np
import pytest

from numgrids import (
    CylindricalGrid,
    Grid,
    PolarGrid,
    SphericalGrid,
    SphericalSurfaceGrid,
    chebyshev,
    colatitude,
    equidistant,
    periodic,
    radial,
)


@pytest.fixture(scope="module")
def ball():
    return SphericalGrid(radial(14, 1.0), colatitude(14), periodic(28))


def spherical_unit_vectors(grid):
    """Cartesian unit vectors in spherical components (r, theta, phi)."""
    _, T, P = grid.meshed_coords
    ex = (np.sin(T) * np.cos(P), np.cos(T) * np.cos(P), -np.sin(P))
    ey = (np.sin(T) * np.sin(P), np.cos(T) * np.sin(P), np.cos(P))
    ez = (np.cos(T), -np.sin(T), np.zeros_like(T))
    return ex, ey, ez


class TestBallVectorCalculus:
    """Vector components change sign across the origin; derivatives must respect that."""

    def test_divergence_of_gradient_is_laplacian(self, ball):
        X, Y, Z = ball.cartesian_coords
        f = np.exp(X) * np.cos(Y) + X ** 2 * Z
        np.testing.assert_allclose(ball.divergence(*ball.gradient(f)), ball.laplacian(f), atol=1e-7)

    def test_divergence_of_cartesian_fields(self, ball):
        X, Y, Z = ball.cartesian_coords
        ex, ey, ez = spherical_unit_vectors(ball)
        # v = (x, 2y, 3z): div = 6
        v = [X * a + 2 * Y * b + 3 * Z * c for a, b, c in zip(ex, ey, ez)]
        np.testing.assert_allclose(ball.divergence(*v), 6.0, atol=1e-10)
        # v = z e_x: div = 0
        np.testing.assert_allclose(ball.divergence(*(Z * a for a in ex)), 0.0, atol=1e-10)

    def test_curl_of_gradient_vanishes(self, ball):
        X, Y, Z = ball.cartesian_coords
        f = np.exp(X) * np.cos(Y) + X ** 2 * Z
        for component in ball.curl(*ball.gradient(f)):
            np.testing.assert_allclose(component, 0.0, atol=1e-8)

    def test_curl_of_cartesian_fields(self, ball):
        X, Y, Z = ball.cartesian_coords
        ex, ey, ez = spherical_unit_vectors(ball)
        # curl(z e_x) = e_y
        for component, expected in zip(ball.curl(*(Z * a for a in ex)), ey):
            np.testing.assert_allclose(component, expected, atol=1e-10)
        # rigid rotation (-y, x, 0): curl = 2 e_z
        v = [-Y * a + X * b for a, b in zip(ex, ey)]
        for component, expected in zip(ball.curl(*v), ez):
            np.testing.assert_allclose(component, 2 * expected, atol=1e-10)

    def test_complex_vector_field(self, ball):
        X, _, _ = ball.cartesian_coords
        ex, _, _ = spherical_unit_vectors(ball)
        div = ball.divergence(*(1j * X * a for a in ex))
        np.testing.assert_allclose(div, 1j, atol=1e-10)


class TestOtherFoldedGrids:
    def test_disk(self):
        disk = PolarGrid(radial(24, 1.0), periodic(32))
        _, P = disk.meshed_coords
        X, Y = disk.cartesian_coords
        ex, ey = (np.cos(P), -np.sin(P)), (np.sin(P), np.cos(P))
        np.testing.assert_allclose(disk.divergence(*(X * a for a in ex)), 1.0, atol=1e-10)
        np.testing.assert_allclose(disk.curl(*(-Y * a + X * b for a, b in zip(ex, ey))), 2.0, atol=1e-10)

    def test_sphere_surface(self):
        sphere = SphericalSurfaceGrid(colatitude(16), periodic(32))
        T, P = sphere.meshed_coords
        f = np.sin(T) * np.cos(P) + np.cos(T) ** 2
        np.testing.assert_allclose(sphere.divergence(*sphere.gradient(f)), sphere.laplacian(f), atol=1e-9)

    def test_cylinder(self):
        cyl = CylindricalGrid(radial(12, 1.0), periodic(16), chebyshev(10, 0, 1))
        X, Y, Z = cyl.cartesian_coords
        f = X ** 2 * Y + Z ** 3 * X
        np.testing.assert_allclose(cyl.divergence(*cyl.gradient(f)), cyl.laplacian(f), atol=1e-8)
        for component in cyl.curl(*cyl.gradient(f)):
            np.testing.assert_allclose(component, 0.0, atol=1e-8)


GRIDS = {
    "cartesian": lambda: Grid(x=chebyshev(14, 0, 1), y=periodic(16)),
    "finite differences": lambda: Grid(x=equidistant(41, 0, 1), y=equidistant(41, 0, 1)),
    "disk": lambda: PolarGrid(radial(16, 1.0), periodic(24)),
    "annulus": lambda: PolarGrid(chebyshev(16, 1, 2), periodic(24)),
    "ball": lambda: SphericalGrid(radial(10, 1.0), colatitude(10), periodic(20)),
    "sphere": lambda: SphericalSurfaceGrid(colatitude(12), periodic(24)),
}


@pytest.mark.parametrize("name", GRIDS)
class TestOperatorForms:
    def field(self, grid):
        if hasattr(grid, "_cartesian_fn"):
            X = grid.cartesian_coords
            return np.exp(0.5 * X[0]) * np.cos(X[1]) + X[0] * X[-1]
        X, Y = grid.meshed_coords
        return np.exp(0.5 * X) * np.cos(Y)

    def test_gradient_operator_matches_gradient(self, name):
        grid = GRIDS[name]()
        f = self.field(grid)
        ops = grid.gradient_operator()
        assert len(ops) == grid.ndims
        for G, expected in zip(ops, grid.gradient(f)):
            np.testing.assert_allclose(G(f), expected, atol=1e-10)

    def test_divergence_operator_matches_divergence(self, name):
        grid = GRIDS[name]()
        v = grid.gradient(self.field(grid))
        div = sum(D(vi) for D, vi in zip(grid.divergence_operator(), v))
        np.testing.assert_allclose(div, grid.divergence(*v), atol=1e-9)

    def test_diffusion_operator(self, name):
        grid = GRIDS[name]()
        f = self.field(grid)
        k = 1 + 0.5 * np.cos(f)
        expected = grid.divergence(*(k * g for g in grid.gradient(f)))
        np.testing.assert_allclose(grid.diffusion_operator(k)(f), expected, atol=1e-8)


def test_diffusion_operator_solves_variable_conductivity():
    # (k u')' = 0, k = 1 + x: u = log(1 + x) / log(2)
    grid = Grid(x=chebyshev(24, 0, 1))
    x = grid.coords
    for k in (1 + x, lambda x: 1 + x):
        u = grid.diffusion_operator(k).solve(0, bcs={"x_low": 0, "x_high": 1})
        np.testing.assert_allclose(u, np.log(1 + x) / np.log(2), atol=1e-12)
    # Constant coefficient: a multiple of the Laplacian
    np.testing.assert_allclose(grid.diffusion_operator(2.0)(x ** 3), 12 * x, atol=1e-9)


def test_diffusion_operator_on_the_ball():
    # div(k grad u) = 1 with k = 1 + r^2 and u(1) = 0 has a radial solution:
    # r^2 k u' = r^3 / 3  =>  u = log((1 + r^2) / 2) / 6
    ball = SphericalGrid(radial(14, 1.0), colatitude(8), periodic(16))
    r = ball.meshed_coords[0]
    u = ball.diffusion_operator(1 + r ** 2).solve(1.0, bcs=0)
    np.testing.assert_allclose(u, np.log((1 + r ** 2) / 2) / 6, atol=1e-9)


def test_parity_argument_is_validated():
    from numgrids.diff import RadialDiff
    disk = PolarGrid(radial(8, 1.0), periodic(8))
    with pytest.raises(ValueError, match="parity"):
        RadialDiff(disk, 1, 0, parity=0)
