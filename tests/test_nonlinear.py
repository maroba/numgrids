import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import brentq

from numgrids import (
    Grid,
    Neumann,
    NonlinearSolverError,
    PolarGrid,
    Robin,
    chebyshev,
    periodic,
    radial,
    solve_nonlinear,
)


def bratu_exact(x, lam, upper=False):
    """Exact solution of u'' + lam e^u = 0, u(0) = u(1) = 0."""
    def f(t):
        return t - np.sqrt(2 * lam) * np.cosh(t / 4)
    theta = brentq(f, 5, 20) if upper else brentq(f, 0, 5)
    return -2 * np.log(np.cosh((x - 0.5) * theta / 2) / np.cosh(theta / 4))


@pytest.fixture
def rod():
    return Grid(x=chebyshev(30, 0, 1))


class TestScalarProblems:

    def test_bratu_finite_difference_jacobian(self, rod):
        x = rod.coords
        u = solve_nonlinear(rod, lambda u: rod.diff(u, 2) + np.exp(u), u0=0, bcs=0)
        npt.assert_allclose(u, bratu_exact(x, 1.0), atol=1e-12)

    def test_bratu_exact_jacobian(self, rod):
        x = rod.coords
        D2 = rod.diff_operator(2)
        u = solve_nonlinear(rod, lambda u: D2(u) + np.exp(u), u0=0, bcs=0,
                            jacobian=lambda u: D2 + np.exp(u))
        npt.assert_allclose(u, bratu_exact(x, 1.0), atol=1e-12)

    def test_initial_guess_selects_branch(self, rod):
        x = rod.coords
        D2 = rod.diff_operator(2)
        u = solve_nonlinear(rod, lambda u: D2(u) + np.exp(u), u0=lambda x: 4 * np.sin(np.pi * x),
                            bcs=0)
        npt.assert_allclose(u, bratu_exact(x, 1.0, upper=True), atol=1e-7)

    def test_nonlinear_diffusion_with_inhomogeneous_dirichlet(self, rod):
        # (k(u) u')' = 0 with k = 1 + u^2: u + u^3 / 3 is linear in x.
        x = rod.coords
        u = solve_nonlinear(rod, lambda u: rod.diff((1 + u ** 2) * rod.diff(u, 1), 1), u0=x,
                            bcs={"x_low": 0, "x_high": 1})
        npt.assert_allclose(u + u ** 3 / 3, 4 / 3 * x, atol=1e-12)

    def test_neumann(self, rod):
        u = solve_nonlinear(rod, lambda u: rod.diff(u, 2) + u - u ** 3, u0=0.8, bcs=Neumann(0))
        npt.assert_allclose(u, 1.0, atol=1e-12)

    def test_robin(self, rod):
        # A monotone nonlinearity (unique solution) with the exact solution u = x^2 + 1:
        # Neumann at x = 0: du/dn = 0; Robin at x = 1: u + du/dn = 2 + 2 = 4.
        x = rod.coords
        u = solve_nonlinear(rod, lambda u: rod.diff(u, 2) - 2 - (u - x ** 2 - 1) ** 3, u0=0,
                            bcs={"x_low": Neumann(0), "x_high": Robin(1, 1, 4)})
        npt.assert_allclose(u, x ** 2 + 1, atol=1e-10)

    def test_full_disk(self):
        # Liouville-Bratu problem on the unit disk, exact solution
        # u = ln(8 mu / lam / (1 + mu r^2)^2) with (1 + mu)^2 = 8 mu / lam.
        lam, mu = 1.0, 3 - 2 * np.sqrt(2)
        grid = PolarGrid(radial(16, 1.0), periodic(16))
        R, _ = grid.meshed_coords
        L = grid.laplacian_operator()
        exact = np.log(8 * mu / lam / (1 + mu * R ** 2) ** 2)
        u = solve_nonlinear(grid, lambda u: L(u) + lam * np.exp(u), 0, bcs=0)
        npt.assert_allclose(u, exact, atol=1e-12)
        u = solve_nonlinear(grid, lambda u: L(u) + lam * np.exp(u), 0, bcs=0,
                            jacobian=lambda u: L + lam * np.exp(u))
        npt.assert_allclose(u, exact, atol=1e-12)

    def test_periodic_without_bcs(self):
        # u'' - u^3 = -sin(x) - sin(x)^3 has the solution sin(x).
        grid = Grid(periodic(32))
        x = grid.coords
        u = solve_nonlinear(grid, lambda u: grid.diff(u, 2) - u ** 3 + np.sin(x) + np.sin(x) ** 3,
                            u0=0)
        npt.assert_allclose(u, np.sin(x), atol=1e-12)

    def test_dense_and_sparse_jacobians(self, rod):
        D2 = rod.diff_operator(2)
        dense = D2.matrix.toarray()
        u1 = solve_nonlinear(rod, lambda u: D2(u) + np.exp(u), 0, bcs=0,
                             jacobian=lambda u: dense + np.diag(np.exp(u)))
        u2 = solve_nonlinear(rod, lambda u: D2(u) + np.exp(u), 0, bcs=0,
                             jacobian=lambda u: D2.matrix + np.diag(np.exp(u)))
        npt.assert_allclose(u1, u2, atol=1e-13)


class TestSystems:

    def test_system_reduces_to_bratu(self, rod):
        x = rod.coords
        D2 = rod.diff_operator(2)
        u, v = solve_nonlinear(rod, lambda u, v: (D2(u) + np.exp(v), v - u), (0, 0),
                               bcs=(0, None))
        npt.assert_allclose(u, bratu_exact(x, 1.0), atol=1e-12)
        npt.assert_allclose(v, u, atol=1e-12)
        u2, _ = solve_nonlinear(rod, lambda u, v: (D2(u) + np.exp(v), v - u), (0, 0),
                                bcs=(0, None),
                                jacobian=lambda u, v: [[D2, np.exp(v)], [-1, 1]])
        npt.assert_allclose(u2, u, atol=1e-12)

    def test_single_bc_spec_applies_to_all_fields(self, rod):
        x = rod.coords
        u, v = solve_nonlinear(rod, lambda u, v: (rod.diff(u, 2) - 2, rod.diff(v, 2) - 2 * u ** 0),
                               (0, 0), bcs={"x_low": 0, "x_high": 1})
        npt.assert_allclose(u, x ** 2, atol=1e-12)
        npt.assert_allclose(v, x ** 2, atol=1e-12)

    def test_block_jacobian_with_empty_row(self, rod):
        D2 = rod.diff_operator(2)
        # The second equation does not depend on anything but v.
        u, v = solve_nonlinear(rod, lambda u, v: (D2(u) - v, v - 2), (0, 0), bcs=(0, None),
                               jacobian=lambda u, v: [[D2, -1], [None, 1]])
        x = rod.coords
        npt.assert_allclose(u, x ** 2 - x, atol=1e-12)


class TestErrors:

    def test_no_convergence(self, rod):
        # The Bratu problem has no solution for lam > 3.51...
        with pytest.raises(NonlinearSolverError, match="did not converge") as info:
            solve_nonlinear(rod, lambda u: rod.diff(u, 2) + 4 * np.exp(u), 0, bcs=0, maxiter=20)
        assert info.value.u.shape == rod.shape
        assert len(info.value.residuals) == 20

    def test_singular_jacobian(self, rod):
        # A Jacobian that is exactly singular:
        with pytest.raises(NonlinearSolverError, match="Singular"):
            solve_nonlinear(rod, lambda u: u ** 2 - 1, 0, jacobian=lambda u: 2 * u)

    def test_bad_residual_and_jacobian(self, rod):
        with pytest.raises(ValueError, match="expected 2"):
            solve_nonlinear(rod, lambda u, v: (u,), (0, 0))
        with pytest.raises(ValueError, match="nested list"):
            solve_nonlinear(rod, lambda u, v: (u, v), (0, 0), jacobian=lambda u, v: [1, 1])
        with pytest.raises(TypeError, match="Jacobian"):
            solve_nonlinear(rod, lambda u: u, 0, jacobian=lambda u: "nonsense")

    def test_verbose(self, rod, capsys):
        solve_nonlinear(rod, lambda u: rod.diff(u, 2) + np.exp(u), 0, bcs=0, verbose=True)
        assert "Newton step 1" in capsys.readouterr().out
