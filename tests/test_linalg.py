"""Linear solvers: method selection, factorization, Fourier decoupling."""

import numpy as np
import pytest

from numgrids import (
    Grid,
    Neumann,
    PolarGrid,
    SphericalGrid,
    chebyshev,
    colatitude,
    equidistant,
    periodic,
    radial,
)
from numgrids.linalg import FourierBlocks, LinearSolver, banded_points, shift_invariant_axes


def poisson_square(n=16):
    grid = Grid(x=chebyshev(n, 0, 1), y=chebyshev(n, 0, 1))
    X, Y = grid.meshed_coords
    exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    return grid, -5 * np.pi ** 2 * exact, exact


class TestMethods:
    @pytest.mark.parametrize("method", ["auto", "dense", "sparse"])
    def test_methods_agree_on_spectral_grid(self, method):
        grid, rhs, exact = poisson_square()
        u = grid.laplacian_operator().solve(rhs, bcs=0, method=method)
        np.testing.assert_allclose(u, exact, atol=1e-8)

    def test_auto_picks_dense_for_spectral_and_sparse_for_finite_differences(self):
        grid, _, _ = poisson_square(10)
        assert grid.laplacian_operator().factorize(bcs=0).method == "dense"
        fd = Grid(x=equidistant(40, 0, 1), y=equidistant(40, 0, 1))
        assert fd.laplacian_operator().factorize(bcs=0).method == "sparse"
        mixed = Grid(x=chebyshev(10, 0, 1), y=equidistant(40, 0, 1))
        assert mixed.laplacian_operator().factorize(bcs=0).method == "sparse"

    def test_auto_picks_fourier_on_the_disk(self):
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        assert disk.laplacian_operator().factorize(bcs=0).method == "fourier"

    def test_unknown_method(self):
        grid, _, _ = poisson_square(8)
        with pytest.raises(ValueError, match="Unknown method"):
            grid.laplacian_operator().solve(1, bcs=0, method="magic")

    def test_fourier_needs_an_invariant_axis(self):
        grid, _, _ = poisson_square(8)
        with pytest.raises(ValueError, match="shift invariant"):
            grid.laplacian_operator().solve(1, bcs=0, method="fourier")

    def test_empty_rows_are_reported(self):
        grid = PolarGrid(chebyshev(10, 0, 1), periodic(8))      # grid points at r = 0
        with pytest.raises(ValueError, match="empty row"):
            grid.laplacian_operator().factorize(bcs={"r_high": 0})

    def test_repr(self):
        grid, _, _ = poisson_square(6)
        assert "method='dense'" in repr(grid.laplacian_operator().factorize(bcs=0))


class TestFactorize:
    def test_many_right_hand_sides(self):
        grid = Grid(x=chebyshev(24, 0, np.pi))
        x = grid.coords
        solver = grid.diff_operator(2).factorize(bcs=0)
        assert isinstance(solver, LinearSolver)
        for k in (1, 2, 3):
            np.testing.assert_allclose(solver(-k ** 2 * np.sin(k * x)), np.sin(k * x), atol=1e-9)
        np.testing.assert_allclose(solver.solve(lambda x: -np.sin(x)), np.sin(x), atol=1e-10)

    def test_boundary_values_are_kept(self):
        grid = Grid(x=chebyshev(16, 0, 1))
        solver = grid.diff_operator(2).factorize(bcs={"x_low": 1, "x_high": Neumann(2)})
        x = grid.coords
        np.testing.assert_allclose(solver(0), 1 + 2 * x, atol=1e-10)
        np.testing.assert_allclose(solver(2), 1 + x ** 2, atol=1e-10)

    def test_implicit_euler(self):
        grid = Grid(x=chebyshev(30, 0, 1))
        dt = 0.01
        step = (1 - dt * grid.laplacian_operator()).factorize(bcs=0)
        u0 = np.sin(np.pi * grid.coords)
        u = u0
        for _ in range(10):
            u = step(u)
        np.testing.assert_allclose(u, (1 + dt * np.pi ** 2) ** -10 * u0, atol=1e-10)

    @pytest.mark.parametrize("method", ["dense", "sparse", "fourier"])
    def test_complex_right_hand_side_and_operator(self, method):
        grid = Grid(x=chebyshev(12, 0, 1), y=periodic(8))
        L = grid.laplacian_operator() - 1
        bcs = {"x_low": 0, "x_high": 0}
        f = grid.evaluate(lambda x, y: np.exp(x) * np.cos(y))
        u = L.solve(f, bcs=bcs, method=method)
        np.testing.assert_allclose(L.solve(1j * f, bcs=bcs, method=method), 1j * u, atol=1e-10)
        v = (L + 2j).solve(f, bcs=bcs, method=method)
        np.testing.assert_allclose(((L + 2j)(v) - f)[grid.interior], 0, atol=1e-8)


class TestFourierDecoupling:
    def test_ball_matches_exact_solution(self):
        ball = SphericalGrid(radial(10, 1.0), colatitude(10), periodic(20))
        solver = ball.laplacian_operator().factorize(bcs=0)
        assert solver.method == "fourier"
        r = ball.meshed_coords[0]
        np.testing.assert_allclose(solver(1.0), (r ** 2 - 1) / 6, atol=1e-11)

    def test_boundary_values_may_depend_on_the_angle(self):
        # Laplace equation on the disk with u = cos(2 phi) on the rim: u = r^2 cos(2 phi)
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        R, P = disk.meshed_coords
        bcs = {"r_high": lambda c: np.cos(2 * c[1])}
        u = disk.laplacian_operator().solve(0, bcs=bcs)
        np.testing.assert_allclose(u, R ** 2 * np.cos(2 * P), atol=1e-11)
        np.testing.assert_allclose(u, disk.laplacian_operator().solve(0, bcs=bcs, method="dense"), atol=1e-11)

    def test_angle_dependent_coefficient_is_detected(self):
        disk = PolarGrid(radial(10, 1.0), periodic(12))
        _, P = disk.meshed_coords
        L = disk.laplacian_operator() - (2 + np.cos(P))
        assert shift_invariant_axes(L.matrix, disk) == []
        solver = L.factorize(bcs=0)
        assert solver.method == "dense"
        u = solver(1.0)
        np.testing.assert_allclose((L(u) - 1)[disk.interior], 0, atol=1e-9)

    def test_two_periodic_axes(self):
        grid = Grid(x=periodic(16), y=periodic(12))
        L = grid.laplacian_operator() - 1
        f = grid.evaluate(lambda x, y: np.sin(x) * np.cos(2 * y))
        solver = L.factorize()
        assert solver.method == "fourier"
        np.testing.assert_allclose(solver(f), -f / 6, atol=1e-12)

    def test_banded_blocks_stay_sparse(self):
        # Finite differences in r, Fourier in phi: one banded system per mode.
        annulus = PolarGrid(equidistant(60, 1, 2), periodic(16))
        L = annulus.laplacian_operator()
        solver = L.factorize(bcs=0)
        assert solver.method == "fourier" and solver._sparse_blocks
        f = annulus.evaluate(lambda r, phi: np.cos(3 * phi) / r)
        np.testing.assert_allclose(solver(f), L.solve(f, bcs=0, method="sparse"), atol=1e-10)
        np.testing.assert_allclose(solver(1j * f), 1j * solver(f), atol=1e-12)

    def test_blocks_reproduce_the_matrix(self):
        disk = PolarGrid(radial(6, 1.0), periodic(8))
        A = disk.laplacian_operator().matrix
        blocks = FourierBlocks(disk, shift_invariant_axes(A, disk))
        dense, sparse = blocks.blocks(A), blocks.sparse_blocks(A)
        rng = np.random.default_rng(0)
        u = rng.standard_normal(disk.size)
        coeffs = blocks.forward(u)
        for m in range(blocks.n_modes):
            np.testing.assert_allclose(sparse[m].toarray(), dense[m], atol=1e-10)
            coeffs[:, m] = dense[m] @ coeffs[:, m]
            assert blocks.conjugate_mode(blocks.conjugate_mode(m)) == m
        np.testing.assert_allclose(blocks.backward(coeffs), A @ u, atol=1e-9)


def test_banded_points():
    fd = Grid(x=equidistant(30, 0, 1), y=periodic(16))
    A = fd.laplacian_operator().matrix
    assert banded_points(A, fd) == 30                 # finite differences in x, Fourier in y
    assert banded_points(A, fd, axes=[1]) == 1
    cheb = Grid(x=chebyshev(12, 0, 1), y=chebyshev(12, 0, 1))
    assert banded_points(cheb.laplacian_operator().matrix, cheb) == 1


class TestSingularSystems:
    """Pure Neumann problems determine the solution up to a constant."""

    @pytest.mark.parametrize("method", ["auto", "fourier", "dense", "sparse"])
    def test_neumann_problem_on_the_disk(self, method):
        disk = PolarGrid(radial(12, 1.0), periodic(16))
        X, Y = disk.cartesian_coords
        u = disk.laplacian_operator().solve(4.0, bcs=Neumann(2.0), method=method)      # u = r^2 + c
        assert np.all(np.isfinite(u))
        exact = X ** 2 + Y ** 2
        np.testing.assert_allclose(u - u.mean(), exact - exact.mean(), atol=1e-8)

    @pytest.mark.parametrize("method", ["auto", "dense", "sparse"])
    def test_neumann_problem_on_an_interval(self, method):
        grid = Grid(x=chebyshev(16, 0, 1))
        x = grid.coords
        u = grid.diff_operator(2).solve(2.0, bcs={"x_low": Neumann(0), "x_high": Neumann(2)}, method=method)
        assert np.all(np.isfinite(u))
        np.testing.assert_allclose(u - u[0], x ** 2, atol=1e-9)

    def test_periodic_poisson_problem(self):
        grid = Grid(x=periodic(32))
        x = grid.coords
        u = grid.diff_operator(2).solve(-np.sin(x))          # u = sin(x) + c
        np.testing.assert_allclose(u - u.mean(), np.sin(x), atol=1e-11)

    def test_exactly_singular_sparse_matrix(self):
        from numgrids import Operator
        grid = Grid(x=equidistant(12, 0, 1))
        A = np.eye(12)
        A[5] = A[4]                                           # two equal rows: exactly singular
        solver = Operator(grid, A).factorize(method="sparse")
        b = np.arange(12.0)
        b[5] = b[4]                                           # consistent right-hand side
        np.testing.assert_allclose(A @ solver(b), b, atol=1e-12)
