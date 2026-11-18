import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import brentq

import numgrids.nonlinear
from numgrids import Grid, Neumann, NonlinearSolverError, Robin, chebyshev, equidistant, solve_nonlinear
from numgrids.nonlinear import ContinuationResult, continuation

# Fold of the Bratu problem u'' + lam e^u = 0, u(0) = u(1) = 0:
# lam_c = 8 / sinh(a)^2 with a = coth(a).
BRATU_FOLD = 3.513830719125161


def bratu_exact(x, lam, upper=False):
    """Exact solution of u'' + lam e^u = 0, u(0) = u(1) = 0."""
    def f(t):
        return t - np.sqrt(2 * lam) * np.cosh(t / 4)
    theta = brentq(f, 4.798, 30) if upper else brentq(f, 0, 4.798)
    return -2 * np.log(np.cosh((x - 0.5) * theta / 2) / np.cosh(theta / 4))


def fold_parameter(branch, k, measure=np.max):
    """Extremum of the parabola lam(m) through the three points around the fold *k*."""
    m = branch.measure(measure)[k - 1:k + 2]
    c = np.polyfit(m, branch.params[k - 1:k + 2], 2)
    return c[2] - c[1] ** 2 / (4 * c[0])


@pytest.fixture
def rod():
    return Grid(x=chebyshev(31, 0, 1))


def bratu(grid):
    """Residual and exact Jacobian of the Bratu problem."""
    D2 = grid.diff_operator(2)
    return (lambda u, lam: D2(u) + lam * np.exp(u)), (lambda u, lam: D2 + lam * np.exp(u))


class TestBratu:

    def test_passes_the_fold(self, rod):
        F, J = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=0.05, max_steps=150)
        assert isinstance(branch, ContinuationResult)
        assert branch.success
        assert "max_steps" in branch.message
        assert len(branch) == 151
        assert branch.params.shape == (151,)
        assert branch.u.shape == (151, 31)

        # Exactly one fold, at the largest parameter of the branch.
        assert len(branch.folds) == 1
        k = branch.folds[0]
        assert k == np.argmax(branch.params)
        npt.assert_allclose(branch.params[k], BRATU_FOLD, atol=1e-3)
        assert branch.params[k] < BRATU_FOLD
        npt.assert_allclose(fold_parameter(branch, k), BRATU_FOLD, atol=2e-4)

        # The size of the solution grows monotonically along the branch,
        # the parameter returns on the upper branch.
        assert np.all(np.diff(branch.measure(np.max)) > 0)
        assert branch.params[-1] < 1.0

        # Every point solves the problem.
        x = rod.coords
        for i in (20, k, 120):
            lam = branch.params[i]
            npt.assert_allclose(branch.u[i], bratu_exact(x, lam, upper=i > k), atol=1e-7)

    def test_lower_and_upper_solution_for_the_same_parameter(self, rod):
        F, J = bratu(rod)
        x = rod.coords

        lower = continuation(rod, F, 0, 0, bcs=0, jacobian=J, param_range=(0, 1))
        assert lower.success
        assert "param_range" in lower.message
        assert lower.params[-1] == 1.0
        assert len(lower.folds) == 0
        npt.assert_allclose(lower.u[-1], bratu_exact(x, 1.0), atol=1e-10)
        npt.assert_allclose(lower.u[-1], solve_nonlinear(rod, lambda u: F(u, 1.0), 0, bcs=0), atol=1e-10)

        # From the lower branch at lam = 2 around the fold, back to lam = 1.
        upper = continuation(rod, F, 0, 2, bcs=0, jacobian=J, param_range=(1, 4), max_steps=200)
        assert upper.success
        assert upper.params[0] == 2.0
        assert upper.params[-1] == 1.0
        assert len(upper.folds) == 1
        npt.assert_allclose(upper.u[0], bratu_exact(x, 2.0), atol=1e-10)
        npt.assert_allclose(upper.u[-1], bratu_exact(x, 1.0, upper=True), atol=1e-7)
        guess = 4 * np.sin(np.pi * x)
        npt.assert_allclose(upper.u[-1], solve_nonlinear(rod, lambda u: F(u, 1.0), guess, bcs=0), atol=1e-8)

    def test_finite_difference_jacobian_gives_the_same_branch(self, rod):
        F, J = bratu(rod)
        exact = continuation(rod, F, 0, 0, bcs=0, jacobian=J, max_steps=50)
        fd = continuation(rod, F, 0, 0, bcs=0, max_steps=50)
        assert len(exact.folds) == 1
        npt.assert_array_equal(fd.folds, exact.folds)
        # The finite-difference Jacobian places the points slightly differently along the branch.
        npt.assert_allclose(fd.params, exact.params, atol=1e-5)
        npt.assert_allclose(fd.u, exact.u, atol=1e-5)

    def test_sparse_bordered_system(self, rod, monkeypatch):
        F, J = bratu(rod)
        dense = continuation(rod, F, 0, 0, bcs=0, jacobian=J, max_steps=50)
        monkeypatch.setattr(numgrids.nonlinear, "_DENSE_LIMIT", 10)
        sparse = continuation(rod, F, 0, 0, bcs=0, jacobian=J, max_steps=50)
        npt.assert_array_equal(sparse.folds, dense.folds)
        npt.assert_allclose(sparse.params, dense.params, atol=1e-8)
        npt.assert_allclose(sparse.u, dense.u, atol=1e-8)

    def test_neumann_condition(self):
        # By symmetry, the left half with u'(1/2) = 0 has the same solutions.
        half = Grid(x=chebyshev(21, 0, 0.5))
        x = half.coords
        F, J = bratu(half)
        branch = continuation(half, F, 0, 2, bcs={"x_low": 0, "x_high": Neumann(0)}, jacobian=J,
                              ds=0.05, param_range=(1, 4), max_steps=200)
        assert branch.params[-1] == 1.0
        k, = branch.folds
        npt.assert_allclose(fold_parameter(branch, k), BRATU_FOLD, atol=2e-4)
        npt.assert_allclose(branch.u[-1], bratu_exact(x, 1.0, upper=True), atol=1e-7)

    def test_large_steps_are_reduced_at_the_fold(self, rod):
        F, J = bratu(rod)
        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=1.0, max_steps=30)
        assert branch.success
        assert len(branch.folds) == 1
        x = rod.coords
        k = branch.folds[0]
        npt.assert_allclose(branch.u[k + 2], bratu_exact(x, branch.params[k + 2], upper=True), atol=1e-7)


    def test_two_dimensions(self):
        # On the unit square, the fold lies at lam_c = 6.808124...
        grid = Grid(x=chebyshev(11, 0, 1), y=chebyshev(11, 0, 1))
        L = grid.laplacian_operator()
        branch = continuation(grid, lambda u, lam: L(u) + lam * np.exp(u), 0, 0, bcs=0,
                              jacobian=lambda u, lam: L + lam * np.exp(u), max_steps=80)
        assert branch.u.shape == (81, 11, 11)
        k, = branch.folds
        npt.assert_allclose(branch.params[k], 6.808124, atol=1e-3)
        assert np.all(np.diff(branch.measure(np.max)) > 0)
        npt.assert_allclose(branch.u[:, 0, :], 0, atol=1e-12)
        npt.assert_allclose(branch.u[:, :, -1], 0, atol=1e-12)


class TestKnownBranches:

    def test_linear_problem(self, rod):
        # u'' + lam = 0, u(0) = u(1) = 0: u = lam w with w = x (1 - x) / 2.
        x = rod.coords
        w = x * (1 - x) / 2
        branch = continuation(rod, lambda u, lam: rod.diff(u, 2) + lam, 0, 0, bcs=0, ds=0.25, max_steps=8)
        assert branch.success
        assert len(branch) == 9
        assert len(branch.folds) == 0
        npt.assert_allclose(branch.u, branch.params[:, None] * w, atol=1e-10)
        # Arclength: ds^2 = mean(du^2) + dlam^2.
        npt.assert_allclose(np.diff(branch.params), 0.25 / np.sqrt(1 + np.mean(w ** 2)), rtol=1e-8)

    def test_direction(self, rod):
        x = rod.coords
        branch = continuation(rod, lambda u, lam: rod.diff(u, 2) + lam, 0, 1, bcs=0, direction=-1,
                              max_steps=30)
        assert np.all(np.diff(branch.params) < 0)
        assert branch.params[0] == 1.0
        assert branch.params[-1] < -1.5
        npt.assert_allclose(branch.u[-1], branch.params[-1] * x * (1 - x) / 2, atol=1e-10)

    def test_param_range(self, rod):
        x = rod.coords
        F = lambda u, lam: rod.diff(u, 2) + lam  # noqa: E731
        up = continuation(rod, F, 0, 0.5, bcs=0, param_range=(-1, 2))
        assert up.success and "param_range" in up.message
        assert up.params[-1] == 2.0
        assert np.all((up.params >= -1) & (up.params <= 2))
        npt.assert_allclose(up.u[-1], x * (1 - x), atol=1e-10)
        down = continuation(rod, F, 0, 0.5, bcs=0, param_range=(-1, 2), direction=-1)
        assert down.params[-1] == -1.0
        npt.assert_allclose(down.u[-1], -x * (1 - x) / 2, atol=1e-10)
        # Starting on the end of the range and leaving it at once: only the start remains.
        stay = continuation(rod, F, 0, 2, bcs=0, param_range=(-1, 2))
        assert stay.success
        npt.assert_array_equal(stay.params, [2.0])

    def test_robin_condition(self, rod):
        # u'' + lam = 0, u(0) = 0, u(1) + u'(1) = 0: u = lam (3 x / 4 - x^2 / 2).
        x = rod.coords
        branch = continuation(rod, lambda u, lam: rod.diff(u, 2) + lam, 0, 0,
                              bcs={"x_low": 0, "x_high": Robin(1, 1, 0)}, param_range=(0, 2))
        assert branch.params[-1] == 2.0
        npt.assert_allclose(branch.u, branch.params[:, None] * (3 * x / 4 - x ** 2 / 2), atol=1e-10)

    @pytest.mark.parametrize("n", [5, 40])
    def test_circle(self, n):
        # u^2 + lam^2 = 1 on every grid point: the branch is the unit circle, and
        # the arclength of the continuation is its arclength, on any grid.
        grid = Grid(x=equidistant(n, 0, 1))
        branch = continuation(grid, lambda u, lam: u ** 2 + lam ** 2 - 1, 1, 0,
                              jacobian=lambda u, lam: 2 * u, ds=0.1, max_steps=80)
        assert branch.success
        npt.assert_allclose(branch.u ** 2 + branch.params[:, None] ** 2, 1, atol=1e-9)
        npt.assert_allclose(branch.u, branch.u[:, :1] * np.ones(n), atol=1e-12)
        angle = np.unwrap(np.arctan2(branch.params, branch.u[:, 0]))
        npt.assert_allclose(np.diff(angle), np.arcsin(0.1), rtol=1e-8)
        # More than one revolution: folds at lam = 1, -1, 1.
        assert len(branch.folds) == 3
        npt.assert_allclose(branch.params[branch.folds], [1, -1, 1], atol=2e-3)

    def test_step_grows_up_to_ds_max(self, rod):
        F = lambda u, lam: rod.diff(u, 2) + lam  # noqa: E731
        branch = continuation(rod, F, 0, 0, bcs=0, ds=0.1, ds_max=0.5, max_steps=10)
        steps = np.diff(branch.params) * np.sqrt(1 + np.mean((rod.coords * (1 - rod.coords) / 2) ** 2))
        npt.assert_allclose(steps[:4], 0.1 * 1.5 ** np.arange(4), rtol=1e-8)
        npt.assert_allclose(steps[4:], 0.5, rtol=1e-8)


class TestSystems:

    def test_system_reduces_to_bratu(self, rod):
        D2 = rod.diff_operator(2)
        F, J = bratu(rod)
        scalar = continuation(rod, F, 0, 0, bcs=0, jacobian=J, max_steps=50)

        def residual(u, v, lam):
            return D2(u) + lam * np.exp(v), v - u

        for jacobian in (None, lambda u, v, lam: [[D2, lam * np.exp(v)], [-1, 1]]):
            branch = continuation(rod, residual, (0, 0), 0, bcs=(0, None), jacobian=jacobian, max_steps=50)
            assert isinstance(branch.u, tuple) and len(branch.u) == 2
            u, v = branch.u
            assert u.shape == v.shape == (51, 31)
            npt.assert_allclose(v, u, atol=1e-10)
            assert len(branch.folds) == 1
            npt.assert_allclose(fold_parameter(branch, branch.folds[0], lambda u, v: u.max()), BRATU_FOLD,
                                atol=2e-3)
            x = rod.coords
            npt.assert_allclose(u[-1], bratu_exact(x, branch.params[-1], upper=True), atol=1e-7)
            # Two identical fields have the same mean square as one: the same arclength.
            # A finite-difference Jacobian places the points less exactly along the
            # branch, most of all at the fold, where the solution changes fastest.
            atol = 1e-5 if jacobian is None else 1e-8
            npt.assert_allclose(branch.params, scalar.params, atol=atol)
            npt.assert_allclose(branch.measure(lambda u, v: np.max(u + v)), 2 * scalar.measure(np.max),
                                atol=atol)


class TestStoppingAndErrors:

    def test_branch_that_ends(self, rod):
        # u = sqrt(1 - lam^2) has no continuation beyond lam = 1 (the residual is NaN).
        branch = continuation(rod, lambda u, lam: u - np.sqrt(1 - lam ** 2), 1, 0, ds_min=1e-3)
        assert not branch.success
        assert "ds_min" in branch.message
        # The branch up to the breakdown is valid.
        assert len(branch) > 10
        assert 0.99 < branch.params[-1] < 1
        npt.assert_allclose(branch.u, np.sqrt(1 - branch.params[:, None] ** 2) * np.ones(31), atol=1e-9)

    def test_corner(self, rod):
        # A branch with a corner: the tangent turns by 90 degrees at lam = 1.
        branch = continuation(rod, lambda u, lam: u - abs(lam - 1), 1, 0, ds_min=1e-3)
        assert not branch.success
        assert branch.params[-1] < 1 + 1e-3

    def test_no_tangent_at_the_start(self, rod):
        # At lam = 1 the branch u = sqrt(1 - lam^2) is vertical and ends.
        branch = continuation(rod, lambda u, lam: u - np.sqrt(1 - lam ** 2), 0, 1)
        assert not branch.success
        assert "starting point" in branch.message
        assert len(branch) == 1
        npt.assert_array_equal(branch.u, np.zeros((1, 31)))

    def test_singular_jacobian_at_the_start(self, rod):
        # At lam = 0, u = 0 the Jacobian 2 u vanishes.
        with pytest.raises(NonlinearSolverError, match="Singular"):
            continuation(rod, lambda u, lam: u ** 2 - lam ** 2, 0, 0, jacobian=lambda u, lam: 2 * u)

    def test_no_first_solution(self, rod):
        F, _ = bratu(rod)
        with pytest.raises(NonlinearSolverError):
            continuation(rod, F, 0, 5, bcs=0)

    def test_callback(self, rod):
        F, J = bratu(rod)
        seen = []

        def callback(u, lam):
            seen.append((u.max(), lam))
            return u.max() > 2

        branch = continuation(rod, F, 0, 0, bcs=0, jacobian=J, callback=callback)
        assert branch.success and "callback" in branch.message
        npt.assert_allclose(seen, np.column_stack([branch.measure(np.max), branch.params]))
        assert np.sum(branch.measure(np.max) > 2) == 1

    def test_max_steps_zero(self, rod):
        F, _ = bratu(rod)
        branch = continuation(rod, F, 0, 1, bcs=0, max_steps=0)
        assert len(branch) == 1
        assert len(branch.folds) == 0
        npt.assert_allclose(branch.u[0], bratu_exact(rod.coords, 1.0), atol=1e-10)

    def test_invalid_arguments(self, rod):
        F, _ = bratu(rod)
        with pytest.raises(ValueError, match="direction"):
            continuation(rod, F, 0, 0, bcs=0, direction=0)
        with pytest.raises(ValueError, match="ds_min"):
            continuation(rod, F, 0, 0, bcs=0, ds=1e-8)
        with pytest.raises(ValueError, match="param_range"):
            continuation(rod, F, 0, 0, bcs=0, param_range=(1, 2))
        with pytest.raises(TypeError, match="real"):
            continuation(rod, F, 1j, 0, bcs=0)

    def test_verbose(self, rod, capsys):
        F, J = bratu(rod)
        continuation(rod, F, 0, 0, bcs=0, jacobian=J, ds=1.0, max_steps=12, verbose=True)
        out = capsys.readouterr().out
        assert "Continuation step 12" in out
        assert "corrector failed" in out


def test_repr_is_concise():
    grid = Grid(x=chebyshev(15, 0, 1))
    branch = continuation(grid, lambda u, lam: grid.diff(u, 2) + lam, u0=0, param=0, bcs=0, ds=0.2, max_steps=3)
    text = repr(branch)
    assert text.startswith("ContinuationResult(4 points") and len(text) < 120
